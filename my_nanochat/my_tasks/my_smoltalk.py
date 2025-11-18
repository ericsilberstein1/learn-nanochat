from datasets import load_dataset
from my_tasks.my_common import MyTask

class MySmolTalk(MyTask):

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ['train', 'test']
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row['messages']
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message['role'] == 'system':
            rest_messages = messages[1:] # optional system message is ok
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2
        for i, message in enumerate(rest_messages):
            expected_role = 'user' if i % 2 == 0 else 'assistant'
            assert message['role'] == expected_role, f"message {i} has wrong role"
            assert isinstance(message['content'], str)
        conversation = {
            "messages": messages
        }
        return conversation