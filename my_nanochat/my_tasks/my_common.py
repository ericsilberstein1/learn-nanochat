# copied from https://github.com/karpathy/nanochat/blob/master/tasks/common.py

class MyTask:

    def __init__(self, start=0, stop=None, step=1):
        assert start >= 0
        assert stop is None or stop >= start
        assert step >= 1
        self.start = start
        self.stop = stop
        self.step = step

    @property
    def eval_type(self):
        # 'generative' or 'categorical'
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step
        assert num >= 0
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int)
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index) # are they all conversations?
        return conversation

    def evaluate(self, problem, completion):
        raise NotImplementedError



def render_mc(question, letters, choices):
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query
