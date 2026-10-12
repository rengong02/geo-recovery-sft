import json

import pytest

from scripts.run_sft import choose_rows, digest_rows, encode_row, gold_only_messages


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        parts = []
        for index, message in enumerate(messages):
            role = message['role']
            content = message.get('content') or ''
            if message.get('tool_calls'):
                content = '<tool_call>' + json.dumps(message['tool_calls']) + '</tool_call>'
            if role == 'assistant' and index == len(messages) - 1:
                content = '<think>\n\n</think>\n\n' + content
            parts.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')
        self.text = ''.join(parts)
        return self.text

    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(character) for character in text],
                'offset_mapping': [(index, index + 1) for index in range(len(text))]}


def sample():
    messages = [{'role': 'system', 'content': 'tools'},
                {'role': 'user', 'content': 'question'}]
    steps = []
    for name, failed in [('load_synthetic_table', False), ('merge_dataframes', False),
                         ('merge_dataframes', False)]:
        call_id = f'call_{len(steps) + 1}'
        messages.extend([
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': call_id, 'type': 'function',
                 'function': {'name': name, 'arguments': {'case': call_id}}}]},
            {'role': 'tool', 'tool_call_id': call_id, 'content': 'observation ' + call_id}])
        steps.append({'tool': name, 'failed': failed, 'seconds': 0.1})
    messages.append({'role': 'assistant', 'content': '{"status":"done"}'})
    return {'sample_id': 'synthetic', 'messages': messages, 'steps': steps, 'tools': []}


def test_mask_excludes_headers_and_all_context_tokens():
    tokenizer = CharacterTokenizer()
    row = sample()
    encoded = encode_row(tokenizer, row, 'gold_only', 10000)
    text = tokenizer.text
    expected = [-100] * len(text)
    start = 0
    for _segment in range(encoded['assistant_segments']):
        start = text.index('<|im_start|>assistant\n', start) + len('<|im_start|>assistant\n')
        end = text.index('<|im_end|>', start) + len('<|im_end|>')
        expected[start:end] = [ord(character) for character in text[start:end]]
        start = end
    assert encoded['labels'] == expected
    assert encoded['assistant_segments'] == 3


def test_gold_only_excludes_silent_wrong_join_as_well_as_error_calls():
    row = sample()
    before = json.dumps(row, sort_keys=True)
    messages = gold_only_messages(row)
    assert [message['tool_calls'][0]['id'] for message in messages
            if message.get('tool_calls')] == ['call_1', 'call_3']
    assert json.dumps(row, sort_keys=True) == before


def test_recovery_masks_failed_tool_call_but_keeps_recovery_supervision():
    tokenizer = CharacterTokenizer()
    row = sample()
    row['steps'][1]['failed'] = True

    encoded = encode_row(tokenizer, row, 'recovery', 10000)
    text = tokenizer.text
    assistant_spans = []
    start = 0
    marker = '<|im_start|>assistant\n'
    while True:
        marker_start = text.find(marker, start)
        if marker_start < 0:
            break
        span_start = marker_start + len(marker)
        span_end = text.index('<|im_end|>', span_start) + len('<|im_end|>')
        assistant_spans.append((span_start, span_end))
        start = span_end

    labelled = []
    for span_start, span_end in assistant_spans:
        labelled.append(any(
            span_start <= token_start < token_end <= span_end
            and label != -100
            for label, (token_start, token_end) in zip(
                encoded['labels'],
                tokenizer(text)['offset_mapping'],
            )
        ))

    assert labelled == [True, False, True, True]


def test_encoding_rejects_truncation_instead_of_dropping_terminal_labels():
    with pytest.raises(ValueError, match='over max_length'):
        encode_row(CharacterTokenizer(), sample(), 'gold_only', 10)


def test_subset_digest_ignores_nonsemantic_timings():
    first = sample()
    second = sample()
    second['steps'][0]['seconds'] = 3.0
    assert digest_rows([first]) == digest_rows([second])
    second['messages'][-1]['content'] = '{"status":"unknown"}'
    assert digest_rows([first]) != digest_rows([second])


def test_probe_selection_is_reproducible_and_never_mixes_splits():
    rows = [{'sample_id': 'train'}, {'sample_id': 'validation'}, {'sample_id': 'test'}]
    split = {'assignments': {row['sample_id']: row['sample_id'] for row in rows}}
    assert choose_rows(rows, split, 'validation', 0, 7) == [rows[1]]
    assert choose_rows(rows, split, 'train', 1, 7) == [rows[0]]


def test_negative_subset_limit_is_rejected():
    with pytest.raises(ValueError, match='limit'):
        choose_rows([{'sample_id': 'train'}], {'assignments': {'train': 'train'}},
                    'train', -1, 7)
