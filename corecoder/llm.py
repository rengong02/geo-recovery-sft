"""LLM provider layer - thin wrapper over OpenAI-compatible APIs.

Since most providers (DeepSeek, Qwen, Kimi, GLM, Ollama, etc.) expose an
OpenAI-compatible endpoint, we just use the openai SDK directly.  Switch
provider by changing OPENAI_BASE_URL + OPENAI_API_KEY. That's it.

For providers that are NOT OpenAI-compatible (AWS Bedrock, Google Vertex,
etc.), use the LiteLLM backend which routes to 100+ providers through a
single unified interface. Set CORECODER_PROVIDER=litellm.
"""

import contextlib
import json
import time
from dataclasses import dataclass, field

from openai import APIConnectionError, APIError, APITimeoutError, BadRequestError, OpenAI, RateLimitError


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str = ""
    #: The chain-of-thought a thinking model streamed alongside the answer.
    reasoning: str = ""
    #: Whether the provider needs `reasoning_content` present on a turn that
    #: carries tool calls *even when that turn produced none*.
    reasoning_round_trip: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def message(self) -> dict:
        """Convert to OpenAI message format for appending to history."""
        msg: dict = {"role": "assistant", "content": self.content or None}
        # DeepSeek's thinking mode rejected a follow-up with a 400 -- "The
        # `reasoning_content` in the thinking mode must be passed back to the
        # API" -- which killed runs seven tasks in. Measured against the live
        # API rather than guessed, because the obvious reading is wrong:
        #
        #   tool-call turn, reasoning non-empty  -> 200
        #   tool-call turn, reasoning ""         -> 200
        #   tool-call turn, key absent           -> 400
        #   turn without tool calls, either way  -> 200
        #
        # So the key must be *present* on every tool-call turn, empty or not;
        # "only send it when we have something" still fails, whenever the model
        # answers without thinking. `reasoning_round_trip` is set only for a
        # provider that has actually streamed reasoning -- an OpenAI-compatible
        # endpoint that does not know the field rejects unknown message keys,
        # and those never set it.
        if self.reasoning or (self.tool_calls and self.reasoning_round_trip):
            msg["reasoning_content"] = self.reasoning
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


# pricing per million tokens: (input, output)
# sources: openai.com/api/pricing, api-docs.deepseek.com, platform.claude.com,
#          platform.moonshot.ai, alibabacloud.com/help/en/model-studio
_PRICING = {
    # OpenAI - current flagships
    "gpt-5.5": (5, 30),
    "gpt-5.4": (2.5, 15),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    "o4-mini": (1.1, 4.4),
    # OpenAI - previous gen (still widely used)
    "gpt-4.1": (2, 8),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o": (2.5, 10),
    "gpt-4o-mini": (0.15, 0.6),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # Anthropic Claude
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-4-6": (3, 15),
    "claude-haiku-4-5": (1, 5),
    # Alibaba Qwen
    "qwen3-max": (0.78, 3.9),
    "qwen3-plus": (0.26, 0.78),
    "qwen-max": (0.78, 3.9),
    # Moonshot Kimi
    "kimi-k3": (3, 15),
    "kimi-k2.5": (0.6, 3),
}


def _adapt_rejected_param(params: dict, exc: BadRequestError) -> bool:
    """Translate or drop one parameter a provider rejected with a 400.

    Newer OpenAI models (gpt-5, o-series) accept only ``max_completion_tokens``
    and the default temperature; the 400 message quotes the offender
    (``'max_tokens'``). Quoted matching is load-bearing: the rejection text
    for ``max_completion_tokens`` must not re-trigger the translation.
    Returns False when nothing recognizable was rejected.
    """
    message = str(exc).lower()
    if "'max_tokens'" in message and "max_tokens" in params:
        params["max_completion_tokens"] = params.pop("max_tokens")
        return True
    if "'temperature'" in message and "temperature" in params:
        params.pop("temperature")
        return True
    return False


class RequestBudgetExceeded(TimeoutError):
    """A request and its retries used up the time they were given.

    Deliberately not the SDK's `APITimeoutError`: that one means the provider
    timed out on us, and this one means *we* stopped waiting. The Agent turns it
    into a deadline stop when the budget came from the task's deadline, so a
    merely slow task is reported as truncated rather than as a crash -- the two
    call for different responses, and conflating them hides which happened.
    """


REQUEST_BUDGET_MESSAGE = (
    "the time allotted to this request -- including every retry and every "
    "backoff -- is used up"
)


def _remaining(deadline_at: float | None) -> float | None:
    """Seconds left before an absolute deadline, or None when unbounded."""
    return None if deadline_at is None else deadline_at - time.monotonic()


def _wait_within(seconds: float, deadline_at: float | None) -> None:
    """Sleep, unless the wait cannot finish before the deadline.

    Sleeping first and checking afterwards is the same mistake one level down:
    the budget is spent either way, and the caller waits to discover it. A
    backoff that does not fit is not worth starting.
    """
    left = _remaining(deadline_at)
    if left is not None and left <= seconds:
        raise RequestBudgetExceeded(REQUEST_BUDGET_MESSAGE)
    time.sleep(seconds)


class LLM:
    #: Latched once this provider has streamed any chain-of-thought. It is how a
    #: turn carrying tool calls knows to send `reasoning_content` back even when
    #: *that* turn produced none. A provider that never reasons leaves it False
    #: and its messages keep the shape it expects. See `LLMResponse.message`.
    #:
    #: A class attribute, not set in `__init__`: `LiteLLM` has its own
    #: `__init__` and would otherwise raise AttributeError from `_drain`.
    saw_reasoning: bool = False

    #: Seconds a single request may take before it is abandoned. A watcher that
    #: consumes a stream gets no signal from a stalled provider, so without a
    #: bound here a wedged stream blocks the whole run: one was observed sitting
    #: from 00:18 to 01:14 and taking every later sample with it.
    DEFAULT_TIMEOUT = 300.0

    #: Declared at class level, not only in `__init__`. `LiteLLM` writes its own
    #: `__init__` and skips this one, so anything set only here is missing on it
    #: -- which has now caused two breakages (`saw_reasoning` was the first).
    #: `test_llm_subclasses.py` keeps a list of these and fails if one is added
    #: to this class without being declared here.
    timeout: float | None = DEFAULT_TIMEOUT

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ):
        self.model = model
        self.timeout = self.DEFAULT_TIMEOUT if timeout is None else timeout
        # `max_retries=0` on purpose. The SDK retries transient failures by
        # itself, and this class retries them again, so the two multiply: three
        # attempts here over three there, each allowed the full client timeout,
        # is up to ninety minutes for what the caller asked to be one request.
        # One retry policy, in `_call_with_retry`, where the budget is visible.
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             timeout=self.timeout or None, max_retries=0)
        self.extra = kwargs  # temperature, max_tokens, etc.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @property
    def estimated_cost(self) -> float | None:
        """Rough cost estimate in USD. Returns None if model not in pricing table."""
        pricing = _PRICING.get(self.model)
        if not pricing:
            return None
        input_rate, output_rate = pricing
        return (
            self.total_prompt_tokens * input_rate / 1_000_000
            + self.total_completion_tokens * output_rate / 1_000_000
        )

    def _reasoning_complete(self, messages: list[dict]) -> list[dict]:
        """Give every assistant turn a `reasoning_content` key, in thinking mode.

        This runs at the request boundary rather than on each response because
        not every assistant message comes from a response -- context compression
        inserts a plain one when it summarizes old turns, and that one was the
        actual trigger of a run-killing 400.

        The rule was measured against the live API, and the obvious reading of
        it is wrong twice over. What the measurements say:

            conversation ends with a user message, assistant has no key   -> 200
            conversation ends with a tool result, assistant has no key    -> 400
            same, but the key is present and empty                        -> 200
            same, but that assistant message is removed                   -> 200

        So it is not only tool-call turns (the first fix), and not only turns
        that produced reasoning (the second). Once a conversation continues
        after a tool result, *every* assistant turn needs the key, empty or not.

        Copies rather than mutates: the caller's history is reused next round,
        and a plain assistant message should stay plain in the transcript.
        A provider that has never streamed reasoning is returned untouched --
        OpenAI-compatible endpoints reject message keys they do not know.
        """
        if not self.saw_reasoning:
            return messages
        out: list[dict] = []
        for message in messages:
            if message.get("role") != "assistant" or "reasoning_content" in message:
                out.append(message)
            else:
                out.append({**message, "reasoning_content": ""})
        return out

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
        on_reasoning=None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Send messages, stream back response, handle tool calls.

        on_reasoning, when given, receives reasoning_content deltas from
        thinking models (deepseek-reasoner, kimi-k2-thinking, ...).

        `timeout` bounds the *whole* call including retries, and tightens -- it
        never loosens -- the client's own. The Agent passes whatever is left of
        the task deadline, which is what turns `Limits.deadline_s` from a
        between-rounds check into something a stalled request cannot outlive.
        """
        budget = self.timeout
        if timeout is not None:
            # Tighten, never loosen: a caller may ask for less than the client's
            # allowance but not for more of it.
            budget = timeout if budget is None else min(timeout, budget)
        params: dict = {
            "model": self.model,
            "messages": self._reasoning_complete(messages),
            "stream": True,
            **self.extra,
        }
        if tools:
            params["tools"] = tools

        # Provider-dialect fallbacks. A 400 that quotes a parameter is adapted
        # (newer OpenAI models take max_completion_tokens, default temperature
        # only); a 400 that names nothing drops stream_options once, matching
        # the previous single-fallback behavior for servers that reject the
        # OpenAI extension silently. The loop is bounded by the param set, and
        # an unrecognized 400 re-raises instead of doubling _call_with_retry's
        # exhausted retries. LiteLLM never lands here: drop_params strips
        # unsupported keys on that path.
        params["stream_options"] = {"include_usage": True}
        # One absolute deadline for the whole method, computed here and shared
        # by every layer below. Passing a *duration* down let each of the three
        # outer attempts start a fresh allowance, so a request bounded at 100ms
        # could still run for three seconds and issue nine requests.
        deadline_at = None if budget is None else time.monotonic() + budget
        last_err = None
        for attempt in range(3):
            while True:
                try:
                    stream = self._call_with_retry(params, deadline_at=deadline_at)
                    break
                except BadRequestError as e:
                    if _adapt_rejected_param(params, e):
                        continue
                    if "stream_options" not in params:
                        raise
                    params.pop("stream_options")
            try:
                return self._drain(stream, on_token, on_reasoning, deadline_at=deadline_at)
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_err = e
            except APIError as e:
                # retry 5xx mid-stream drops but not 4xx
                status_code = getattr(e, "status_code", None)
                if not status_code or status_code < 500:
                    raise
                last_err = e
            # The stream died partway through. Nothing has been executed yet,
            # so re-issuing the whole request is safe; whatever partial text
            # on_token already showed is simply regenerated. Create-time
            # transients stay inside _call_with_retry and do not land here.
            if attempt < 2:
                _wait_within(2**attempt, deadline_at)
        raise last_err

    @staticmethod
    def _abandon(stream) -> None:
        """Tell the transport we are done with a stream we are giving up on.

        Best effort: whatever is left of the response is not wanted, and holding
        the connection open would keep a slot busy for nobody.
        """
        close = getattr(stream, "close", None)
        if callable(close):
            # Closing must never mask the reason we are closing: whatever the
            # transport says here, the caller is already getting the real error.
            with contextlib.suppress(Exception):
                close()

    def _drain(self, stream, on_token, on_reasoning, deadline_at: float | None = None) -> LLMResponse:
        """Consume one stream into an LLMResponse, stopping at `deadline_at`.

        Checking here, not only around the request, is what makes the bound
        total rather than per-attempt: a provider that keeps trickling chunks
        never trips a read timeout, so a stream can cross the deadline and still
        hand back a perfectly normal-looking response. The reply is then
        truncated without anything saying so, which is worse than a refusal.

        **What the wall clock is actually bounded by** -- and it is *not* the
        deadline on the nose. Two mechanisms, neither sufficient alone:

        * this check, at every chunk boundary. Bounds a stream that keeps
          producing chunks, but only from the moment a chunk arrives: if the
          deadline passes at t and the next chunk lands at t+g, the overrun is
          g. The loop cannot see the clock while it is parked inside `next()`.
        * the per-attempt request timeout -- `min(self.timeout, remaining)`,
          handed to httpx, which applies it **per read operation**, not to the
          response as a whole. Bounds a stream that stops producing chunks at
          all, because the allowance is what was left when that attempt
          started, not a fresh full one.

        So the honest bound is **deadline + the longest gap between chunks**,
        capped by that attempt's read timeout. A shorter deadline does not
        shrink it; a smaller `timeout` does. Do not write "never exceeds the
        deadline" anywhere -- it is close to true, and false.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_map: dict[int, dict] = {}  # index -> {id, name, arguments_str}
        prompt_tok = 0
        completion_tok = 0

        for chunk in stream:
            left = _remaining(deadline_at)
            if left is not None and left <= 0:
                self._abandon(stream)
                raise RequestBudgetExceeded(REQUEST_BUDGET_MESSAGE)
            # usage info comes in the final chunk; getattr-style reads stay safe
            # across OpenAI SDK objects and litellm's provider-varying shapes
            usage = getattr(chunk, "usage", None)
            if usage:
                # some providers send usage with null fields; coerce to 0 so the
                # running totals below don't blow up on int + None
                prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
                completion_tok = getattr(usage, "completion_tokens", 0) or 0

            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta

            # thinking models stream their chain-of-thought separately; never
            # mix it into the reply, but do keep it. reasoning_content is the
            # DeepSeek/Kimi dialect, reasoning is what aggregators like
            # OpenRouter normalize it to.
            reasoning = getattr(delta, "reasoning_content", None) or getattr(
                delta, "reasoning", None)
            if reasoning:
                reasoning_parts.append(reasoning)
                if on_reasoning:
                    on_reasoning(reasoning)

            # accumulate text
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                if on_token:
                    on_token(delta.content)

            # accumulate tool calls across chunks
            if getattr(delta, "tool_calls", None):
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_map[idx]["args"] += tc_delta.function.arguments

        # parse accumulated tool calls
        parsed: list[ToolCall] = []
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            try:
                args = json.loads(raw["args"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            parsed.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))

        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok
        # Latched, not per-response: one reasoning turn marks the provider as
        # thinking, and from then on *every* tool-call turn must carry the key.
        if reasoning_parts:
            self.saw_reasoning = True

        return LLMResponse(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            reasoning_round_trip=self.saw_reasoning,
            tool_calls=parsed,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
        )

    def _call_with_retry(self, params: dict, max_retries: int = 3,
                         deadline_at: float | None = None):
        """Retry on transient errors with exponential backoff, before `deadline_at`.

        **An absolute monotonic time, not a duration.** A duration has to be
        re-based by every caller that loops around this method, and a caller
        that forgets hands each attempt a fresh allowance -- which is exactly
        how a "bounded" request still took three times its timeout. An absolute
        deadline cannot be re-based by accident.
        """
        for attempt in range(max_retries):
            left = _remaining(deadline_at)
            if left is not None and left <= 0:
                raise RequestBudgetExceeded(REQUEST_BUDGET_MESSAGE)
            attempt_params = params if left is None else {**params, "timeout": left}
            try:
                return self.client.chat.completions.create(**attempt_params)
            except (RateLimitError, APITimeoutError, APIConnectionError):
                if attempt == max_retries - 1:
                    raise
                _wait_within(2 ** attempt, deadline_at)
            except APIError as e:
                # retry 5xx server errors but not 4xx; base APIError has no status_code so read it defensively
                status_code = getattr(e, "status_code", None)
                if status_code and status_code >= 500 and attempt < max_retries - 1:
                    _wait_within(2 ** attempt, deadline_at)
                else:
                    raise


class LiteLLM(LLM):
    """LLM backend via LiteLLM, supporting 100+ providers.

    Use this when your target provider is NOT OpenAI-compatible
    (AWS Bedrock, Google Vertex, Cohere, etc.) or when you want
    a single interface to switch between any provider by changing
    the model string.

    Set CORECODER_PROVIDER=litellm and use LiteLLM model strings
    like ``anthropic/claude-3-haiku``, ``bedrock/anthropic.claude-v2``,
    ``vertex_ai/gemini-pro``, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ):
        # skip LLM.__init__ which creates an OpenAI client
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = self.DEFAULT_TIMEOUT if timeout is None else timeout
        self.extra = kwargs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _call_with_retry(self, params: dict, max_retries: int = 3,
                         deadline_at: float | None = None):
        """Retry on transient errors with exponential backoff via litellm.

        `deadline_at` is the same absolute deadline the OpenAI-compatible path
        takes -- the two overrides have to agree on the contract, or a caller
        that passes to one silently breaks against the other.
        """
        params["drop_params"] = True
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url

        left = _remaining(deadline_at)
        if left is not None and left <= 0:
            # Checked before the import: a budget that is already spent should
            # not cost a heavy import on the way out.
            raise RequestBudgetExceeded(REQUEST_BUDGET_MESSAGE)

        import litellm

        for attempt in range(max_retries):
            left = _remaining(deadline_at)
            if left is not None and left <= 0:
                raise RequestBudgetExceeded(REQUEST_BUDGET_MESSAGE)
            attempt_params = params if left is None else {**params, "timeout": left}
            try:
                return litellm.completion(**attempt_params)
            except Exception as e:
                err = str(e).lower()
                is_transient = any(
                    kw in err
                    for kw in ["rate_limit", "timeout", "connection", "502", "503", "529"]
                )
                is_server = any(kw in err for kw in ["500", "502", "503", "504"])
                if (is_transient or is_server) and attempt < max_retries - 1:
                    _wait_within(2 ** attempt, deadline_at)
                else:
                    raise
