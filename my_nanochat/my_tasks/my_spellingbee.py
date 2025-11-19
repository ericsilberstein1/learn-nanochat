from my_tasks.my_common import MyTask
from my_nanochat.my_common import download_file_with_lock
import random
import re

WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
TEST_RANDOM_SEED_OFFSET = 10_000_000 # much bigger than the ~370,000 words in the list

class MySimpleSpelling(MyTask):
    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ['train', 'test']
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split('/')[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        rng = random.Random(42)
        rng.shuffle(words)
        self.words = words

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)
        word = rng.choice(self.words)
        word_letters = ','.join(list(word))
        messages = [
            {'role': 'user', 'content': f"Spell the word: {word}"},
            {'role': 'assistant', 'content': f"{word}:{word_letters}"}
        ]
        conversation = {
            'messages': messages
        }
        return conversation

LETTERS = "abcdefghijklmnopqrstuvwxyz"

ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    match = ANSWER_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None

USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    # Spanish
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    # Chinese (Simplified)
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    # Korean
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    # French
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    # German
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    # Japanese
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]

class MySpellingBee(MyTask):

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ['train', 'test']
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split('/')[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        self.words = words

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)
        
        word = rng.choice(self.words)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)

        count = word.count(letter)

        template = rng.choice(USER_MSG_TEMPLATES)
        if rng.random() < 0.3:
            template = template.lower()
        quote_options = ['', '"', "'"]
        letter_quote = rng.choice(quote_options)
        word_quote = rng.choice(quote_options)
        letter_wrapped = f"{letter_quote}{letter}{letter_quote}"
        word_wrapped = f"{word_quote}{word}{word_quote}"
        user_msg = template.format(letter=letter_wrapped, word=word_wrapped)
        if rng.random() < 0.5:
            user_msg += "?"

        assistant_parts = []
        word_letters = ",".join(list(word))
        manual_text = f"""We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.

First spell the word out:
{word}:{word_letters}

Then count the occurrences of '{letter}':
"""
        # see his interesting comment here about ways to improve this and what should emerge from RL
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"

        manual_text += f"\nThis gives us {running_count}."
        assistant_parts.append({'type': 'text', 'text': manual_text})

        assistant_parts.append({'type': 'text', 'text': "\n\nLet me double check this using Python:\n\n"})

        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts.append({'type': 'python', 'text': python_expr})
        assistant_parts.append({'type': 'python_output', 'text': str(count)})
        assistant_parts.append({'type': 'text', 'text': f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"})

        messages = [
            {'role': 'user', 'content': user_msg},
            {'role': 'assistant', 'content': assistant_parts}
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
    



