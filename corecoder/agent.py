"""Core agent loop.

This is the heart of CoreCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

import concurrent.futures
import inspect
import time

from .context import ContextManager
from .limits import DEADLINE, LimitExceeded, Limits
from .llm import LLM, RequestBudgetExceeded
from .permissions import Permission
from .prompt import PLAN_MODE_PROMPT, system_prompt
from .tool_errors import call_key, is_error_result, with_repeat_marker
from .tools import ALL_TOOLS
from .tools.agent import AgentTool
from .tools.base import Tool
from .tools.todo import TodoWriteTool

DEFAULT_MAX_ROUNDS = 50


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int | None = None,
        permission=None,
        hooks=None,
        limits: Limits | None = None,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self.permission = permission
        self.hooks = hooks
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        #: Identical calls that came back as failures, by `call_key`. Diagnostics
        #: only -- it never changes the score, and it never triggers a retry.
        self._failure_repeats: dict[str, int] = {}
        #: Absolute monotonic time this turn must end by, set per `chat()`.
        #: Tools read it so a slow one cannot outlive the run that asked for it;
        #: without it an MCP call used its own fixed timeout and a task bounded
        #: at five seconds could sit in a tool for a minute.
        self._turn_deadline: float | None = None
        #: (tool, seconds past the deadline, that tool's `deadline_mode`) for
        #: every call that came back late. Diagnostics, never scored -- and the
        #: only honest way to report a cooperative tool, which cannot be stopped
        #: mid-call and must therefore be measured rather than promised.
        self._deadline_overruns: list[tuple[str, float, str]] = []
        # `max_rounds` predates `limits` and stays as the shorthand for it, so
        # existing callers keep working and there is one source of truth. The
        # default is None rather than 50 so that "not passed" and "passed 50"
        # are distinguishable -- otherwise `max_rounds=50, limits=...` would be
        # silently resolved in favour of `limits` with no way to notice.
        if limits is not None and max_rounds is not None:
            raise ValueError(
                "pass either limits= or max_rounds=, not both "
                f"(got limits.max_rounds={limits.max_rounds}, max_rounds={max_rounds})"
            )
        self.limits = (
            limits if limits is not None
            else Limits(max_rounds=max_rounds if max_rounds is not None else DEFAULT_MAX_ROUNDS)
        )
        self.max_rounds = self.limits.max_rounds
        self._system = system_prompt(self.tools)
        self.plan_mode = False  # toggled by /plan; while on, mutating tools are refused

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

        self._todo = next((t for t in self.tools if isinstance(t, TodoWriteTool)), None)

    def _full_messages(self) -> list[dict]:
        system = self._system
        # re-injected every round, like the task list below, so a toggle made
        # between turns takes effect on the very next request
        if self.plan_mode:
            system += "\n\n" + PLAN_MODE_PROMPT
        # the task list is re-injected every round, so the model always sees the
        # current state rather than a stale copy buried in old tool results
        if self._todo is not None:
            rendered = self._todo.render()
            if rendered:
                system += "\n\n# Current task list\n" + rendered
        return [{"role": "system", "content": system}] + self.messages

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def _tokens_used(self) -> int:
        """Lifetime tokens on this LLM client; callers measure deltas."""
        return self.llm.total_prompt_tokens + self.llm.total_completion_tokens

    def _cost_delta(self, base: float | None) -> float | None:
        """USD spent since `base`, or None when the model has no price listed.

        `estimated_cost` is a property on `LLM`, not a method -- reading it with
        parentheses raises `TypeError: 'float' object is not callable`.
        """
        now = self.llm.estimated_cost
        if now is None:
            return None
        return now - (base or 0.0)

    def chat(self, user_input: str, on_token=None, on_tool=None, on_reasoning=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds.

        Raises `LimitExceeded` when the turn hits a configured budget. The turn
        did not finish in that case, and the exception names the limit that
        tripped -- a caller recording results needs that distinction, because
        "stopped early" and "finished" are different outcomes that a return
        string would blur together.

        Budgets are measured per call: tokens, cost and wall-clock are deltas
        from the start of this turn, so one Agent can serve many turns.
        """
        self.messages.append({"role": "user", "content": user_input})
        self.context.maybe_compress(self.messages, self.llm)

        started_at = time.monotonic()
        # Computed once per turn and read by every tool call in it -- including
        # the ones a pool worker runs, which is why it is on the instance rather
        # than threaded through the call.
        self._turn_deadline = (
            None if self.limits.deadline_s is None else started_at + self.limits.deadline_s
        )
        tokens_at_start = self._tokens_used()
        cost_at_start = self.llm.estimated_cost
        tool_calls = 0
        rounds = 0

        while True:
            rounds += 1
            self.limits.check_rounds(rounds)
            self.limits.check_deadline(started_at)

            try:
                resp = self.llm.chat(
                    messages=self._full_messages(),
                    tools=self._tool_schemas(),
                    on_token=on_token,
                    on_reasoning=on_reasoning,
                    # Whatever is left of the task's deadline. Until this was
                    # passed, `deadline_s` was checked only *between* rounds, so
                    # one stalled stream could outlive the run that asked for it.
                    timeout=self.limits.remaining_seconds(started_at),
                )
            except RequestBudgetExceeded as e:
                if self.limits.deadline_s is None:
                    raise
                # The budget came from this task's deadline, so this is the
                # deadline being reached, not the run falling over. Reported as
                # truncated, which is a normal outcome here.
                raise LimitExceeded(
                    DEADLINE, self.limits.deadline_s, time.monotonic() - started_at
                ) from e

            # Record what the model said *before* checking budgets. Raising with
            # the assistant turn unwritten would leave a user message with no
            # reply, so a caller that caught the error and reused this Agent
            # would carry an invalid transcript into the next turn.
            self.messages.append(resp.message)

            try:
                # usage lands with the response, so token/cost are checked here
                # rather than before the call that spends it
                self.limits.check_tokens(self._tokens_used() - tokens_at_start)
                self.limits.check_cost(self._cost_delta(cost_at_start))
                # a single slow generation can cross the deadline mid-call;
                # this is the first moment we can notice
                self.limits.check_deadline(started_at)

                # no tool calls -> LLM is done, return text
                if not resp.tool_calls:
                    return resp.content

                tool_calls += len(resp.tool_calls)
                self.limits.check_tool_calls(tool_calls)
            except LimitExceeded:
                # the model asked for calls we are not going to run; give each
                # one a reply so the transcript stays a valid sequence
                self._answer_pending_tool_calls(resp.tool_calls)
                raise

            # tool calls -> execute (parallel when multiple, like Claude Code's
            # StreamingToolExecutor which runs independent tools concurrently)

            try:
                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    result = self._pre_hooks(tc) or self._permit(tc)
                    if result is None:
                        result = self._exec_tool(tc)
                        self._post_hooks(tc, result)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                else:
                    # parallel execution for multiple tool calls
                    results = self._exec_tools_parallel(resp.tool_calls, on_tool)
                    for tc, result in zip(resp.tool_calls, results):
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
            except KeyboardInterrupt:
                # Ctrl+C mid-execution would leave the assistant tool_calls
                # message without replies, poisoning the next request; backfill
                self._answer_pending_tool_calls(resp.tool_calls)
                raise

            # compress if tool outputs are big
            self.context.maybe_compress(self.messages, self.llm)

    def _pre_hooks(self, tc) -> str | None:
        """PreToolUse hooks, fired before consent. A string return blocks the
        call and becomes the tool result the model sees; None lets it through."""
        if self.hooks is None:
            return None
        return self.hooks.run_pre(tc.name, tc.arguments)

    def _post_hooks(self, tc, result: str):
        """PostToolUse hooks observe a finished call; they can never block."""
        if self.hooks is not None:
            self.hooks.run_post(tc.name, tc.arguments, result)

    def _permit(self, tc) -> str | None:
        """Consent check for one call. None means go ahead; a string is the
        refusal, returned as the tool result instead of executing."""
        # plan mode outranks consent, even --yes: while it's on nothing mutates
        if self.plan_mode and tc.name not in Permission.READ_ONLY:
            return (
                "Plan mode is on, so this call was refused: plan mode is "
                "read-only. Do not retry it. Keep investigating with the "
                "read-only tools, then present the plan and stop. The user can "
                'approve it by typing "approve", or exit plan mode with /plan.'
            )
        if self.permission is None:
            return None
        return self.permission.check(tc.name, tc.arguments)

    def _note_failure(self, tc, result: str) -> str:
        """Count identical failures, and say so once repeating is pointless.

        Counting only. A framework that silently re-sends the call would make
        "the model repeated itself" and "the harness retried it" one event in
        the record; they are different findings with different fixes. The model
        keeps the decision, it just stops having to make it blind.
        """
        if not is_error_result(result):
            return result
        key = call_key(tc.name, tc.arguments)
        repeats = self._failure_repeats.get(key, 0) + 1
        self._failure_repeats[key] = repeats
        return with_repeat_marker(result, repeats)

    @property
    def failure_repeats(self) -> dict[str, int]:
        """Identical calls that failed, keyed by tool and arguments.

        Reported next to the trajectory rather than inside it: this is what the
        run cost, not what the run did, and the judge must not see it.
        """
        return dict(self._failure_repeats)

    def _exec_tool(self, tc) -> str:
        """Execute a single tool call, returning the result string."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return self._note_failure(tc, f"Error: unknown tool '{tc.name}'")
        # validate arguments first so a TypeError raised *inside* the tool isn't
        # mislabelled as a bad-arguments error from the caller
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return self._note_failure(tc, f"Error: bad arguments for {tc.name}: {e}")
        if self._turn_deadline is not None and time.monotonic() >= self._turn_deadline:
            return self._note_failure(
                tc,
                f"Error executing {tc.name}: not started, the run's deadline had already passed",
            )
        # a tool that blows up gets reported back as text, never kills the loop
        started = time.monotonic()
        try:
            result = tool.execute_with_deadline(self._turn_deadline, **tc.arguments)
        except Exception as e:  # noqa: BLE001
            result = f"Error executing {tc.name}: {e}"
        self._note_overrun(tc, started, getattr(tool, "deadline_mode", "cooperative"))
        return self._note_failure(tc, result)

    def _note_overrun(self, tc, started: float, mode: str) -> None:
        """Record a tool that came back after the turn's deadline.

        A cooperative tool -- an in-process GIS function, say -- cannot be
        stopped mid-call: nothing outside it can interrupt the frame it is in.
        So the honest position is not "tools are bounded" but "tools that can
        bound themselves do, and the ones that cannot are counted". A run whose
        tools overran is still a valid run; a report that does not say so is
        not.
        """
        if self._turn_deadline is None:
            return
        over = time.monotonic() - self._turn_deadline
        if over > 0:
            self._deadline_overruns.append((tc.name, round(over, 3), mode))

    @property
    def deadline_overruns(self) -> list[tuple[str, float, str]]:
        """Calls that returned past the deadline, by tool, seconds late, and mode."""
        return list(self._deadline_overruns)

    def _all_concurrency_safe(self, tool_calls) -> bool:
        """Whether this batch may go to the pool.

        One unsafe tool is enough to serialise the whole batch: its siblings
        have to keep their place in the model's order for it to be correct, and
        a pool gives no such guarantee. Running the batch in order is always
        semantically right, so this costs nothing but speed.
        """
        for tc in tool_calls:
            tool = self._tool_by_name.get(tc.name)
            # an unknown tool cannot be inspected; it fails in `_exec_tool`
            # anyway, and executing that in order is the safe reading
            if tool is None or not getattr(tool, "concurrency_safe", True):
                return False
        return True

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        """Run multiple tool calls concurrently using threads.

        This is inspired by Claude Code's StreamingToolExecutor which starts
        executing tools while the model is still generating.  We simplify to:
        when the model returns N tool calls at once, run them in parallel --
        unless one of them declares itself unsafe to run beside its siblings,
        in which case the batch runs in order instead.
        """
        from .tools.bash import get_tracked_cwd, set_tracked_cwd

        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        # hooks and consent are settled up front on this thread: prompting
        # from pool workers would interleave several prompts on one terminal
        results = [self._pre_hooks(tc) or self._permit(tc) for tc in tool_calls]

        if not self._all_concurrency_safe(tool_calls):
            # In the model's order, on this thread. Same bookkeeping as the pool
            # path below, minus the pool: a skipped call (a hook or the
            # permission layer already answered it) is not executed again, and
            # post-hooks fire only for calls that actually ran.
            for i, tc in enumerate(tool_calls):
                if results[i] is None:
                    results[i] = self._exec_tool(tc)
                    self._post_hooks(tc, results[i])
            return results

        # the tracked cwd is thread-local, so pool workers would otherwise
        # start from the launch directory: hand the session cwd in, and merge
        # any cd back in call order so the batch behaves like a sequence
        session_cwd = get_tracked_cwd()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            def run(i, tc, base_cwd):
                is_bash = tc.name == "bash"
                if is_bash and base_cwd is not None:
                    set_tracked_cwd(base_cwd)
                out = self._exec_tool(tc)
                if not is_bash:
                    return i, out, None
                cwd_after = get_tracked_cwd()
                # only a worker that actually moved the dir reports back; a
                # sibling that merely inherited the base must not revert a cd
                return i, out, cwd_after if cwd_after != base_cwd else None

            futures = {
                i: pool.submit(run, i, tc, session_cwd)
                for i, tc in enumerate(tool_calls)
                if results[i] is None
            }
            for i, future in futures.items():
                i, results[i], worker_cwd = future.result()
                if worker_cwd is not None:
                    session_cwd = worker_cwd
                    set_tracked_cwd(worker_cwd)
        for i in futures:
            self._post_hooks(tool_calls[i], results[i])
        return results

    def _answer_pending_tool_calls(self, tool_calls):
        """Backfill a tool reply for every call that didn't get one.

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()
        # a user-visible reset must not leak the dead conversation's checklist
        # into the next system prompt
        todo = self._tool_by_name.get("todo_write")
        if isinstance(todo, TodoWriteTool):
            todo._tasks = []
