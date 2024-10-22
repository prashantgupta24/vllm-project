import pytest

from tests.tokenization.test_detokenize import TRUTH, _run_incremental_decode
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer


TOKENIZERS = ["mistralai/Pixtral-12B-2409"]

TRUTH += ["THIS IS AN URGENCY"]


@pytest.mark.parametrize("tokenizer_id", TOKENIZERS)
def test_incremental_decode(tokenizer_id):
    tokenizer = MistralTokenizer.from_pretrained(tokenizer_id)
    _run_incremental_decode(
        tokenizer, [1492, 1176, 115679], skip_special_tokens=True, starting_index=0
    )


@pytest.mark.parametrize("tokenizer_id", TOKENIZERS)
@pytest.mark.parametrize("truth", TRUTH)
def test_mistral_tokenizer(tokenizer_id, truth):
    tokenizer = MistralTokenizer.from_pretrained(tokenizer_id)
    all_input_ids = tokenizer.encode(truth)
    generated = truth
    starting_index = 0

    if tokenizer.bos_token_id is not None:
        all_input_ids = [tokenizer.bos_token_id] + all_input_ids
        starting_index += 1
    all_input_ids = all_input_ids + [tokenizer.eos_token_id]

    decoded_text = _run_incremental_decode(
        tokenizer,
        all_input_ids,
        skip_special_tokens=True,
        starting_index=starting_index,
    )

    assert decoded_text == generated

    decoded_text = _run_incremental_decode(
        tokenizer,
        [len(tokenizer)],
        skip_special_tokens=True,
        starting_index=starting_index,
    )

    assert decoded_text == ""
