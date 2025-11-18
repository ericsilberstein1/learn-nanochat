import re
from datasets import load_dataset
from my_tasks.my_common import MyTask

GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None

class MyGSM8K(MyTask):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ['main', 'socratic']
        assert split in ['train', 'test']
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row['question']
        answer = row['answer']
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # calculator tool call
                inner = part[2:-2]
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ''
                assistant_message_parts.append({'type': 'python', 'text': expr})
                assistant_message_parts.append({'type': 'python_output', 'text': result})
            else:
                # regular text between tool calls
                assistant_message_parts.append({'type': 'text', 'text': part})
        messages = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': assistant_message_parts}
        ]
        conversation = {
            'messages': messages
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert isinstance(assistant_response, str)
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == 'assistant'
        assert isinstance(assistant_message['content'], list)
        last_text_part = assistant_message['content'][-1]['text'] # contains final answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float
