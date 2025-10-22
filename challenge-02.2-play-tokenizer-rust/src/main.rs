// Rust version of play tokenzier like ../challenge-01-play-tokenizer/tokenizer.ipynb

use std::collections::HashSet;
use std::collections::HashMap;

type Words = Vec<Vec<String>>; // e.g. [["T", "h", "e"], ["c", "a", "t"]] or [["Th", "e"], ["c", "at"]]
type TokenToId = HashMap<String, usize>;
type IdToToken = HashMap<usize, String>;

#[derive(PartialEq, Eq, Hash, Debug)]
struct Pair {
    first: String,
    second: String,
}

impl Pair {
    fn to_token(&self) -> String {
        format!("{}{}", self.first, self.second)
    }
}

fn generate_pairs(words: &Words) -> HashMap<Pair, u32> {
    let mut pairs: HashMap<Pair, u32> = HashMap::new();
    for word in words {
        for window in word.windows(2) {
            let pair = Pair{first: window[0].to_string(), second: window[1].to_string()};
            *pairs.entry(pair).or_insert(0) += 1;
        }
    }
    pairs
}

// make more idiomatic with .max_by_key() ?
fn find_top_pair(pairs: &HashMap<Pair, u32>) -> &Pair {
    let mut top_frequency: u32 = 0;
    let mut candidate_pair: Option<&Pair> = None;
    for (pair, &frequency) in pairs {
        if frequency > top_frequency {
            top_frequency = frequency;
            candidate_pair = Some(pair);
        }
    }
    candidate_pair.unwrap()
}

fn update_words_with_new_token(words: &mut Words, new_pair: &Pair) {
    for word in words.iter_mut() {
        let mut i: usize = 0;
        while i < word.len() - 1 {
            if (Pair{first: word[i].clone(), second: word[i+1].clone()}) == *new_pair { // ??? why do I need that clone?
                let mut new_word = vec![];
                new_word.extend_from_slice(&word[..i]);
                new_word.push(new_pair.to_token());
                new_word.extend_from_slice(&word[i+2..]);
                *word = new_word;
            }
            i += 1
        }
    }
}

const N_TOKENS: usize = 20;

fn encode_word(token_to_id: &TokenToId, word: &str) -> Vec<usize> {
    let mut encoded: Vec<usize> = vec![];
    let mut unencoded_part: String = word.to_string();
    while unencoded_part.chars().count() > 0 {
        let mut token_id: Option<usize> = None;
        let mut i: usize = unencoded_part.chars().count();
        while token_id.is_none() && i > 0 {
            let substring: String = unencoded_part.chars().take(i).collect();
            token_id = token_to_id.get(&substring).copied(); // why copied ?
            i -= 1;
        }
        if token_id.is_none() {
            token_id = token_to_id.get("<unk>").copied();
        }
        encoded.push(token_id.unwrap());
        unencoded_part = unencoded_part.chars().skip(i+1).collect();
    }
    encoded
}

fn encode(token_to_id: &TokenToId, sentence: &str) -> Vec<usize> {
    let mut encoded: Vec<usize> = vec![];
    for word in sentence.split(' ') {
        encoded.extend(encode_word(token_to_id, word));
        encoded.push(token_to_id[" "]);
    }
    encoded.pop();
    encoded
}

fn decode(id_to_token: &IdToToken, encoded_sentence: &Vec<usize>) -> String {
    encoded_sentence.iter().map(|id| id_to_token[id].as_str()).collect()
}

fn main() {
    let corpus = "The batat and the cat fought over the hat.";
    println!("corpus: {}\n", corpus);

    let mut words: Words = vec![];
    for word in corpus.split(' ') {
        words.push(word.chars().map(|c| c.to_string()).collect());
    }
    println!("initial words: {:?}\n", words);

    let mut tokens: HashSet<String> = HashSet::new();
    for word in &words {
        for c in word {
            tokens.insert(c.to_string());
        }
    }
    println!("initial tokens: {:?}\n", tokens);


    while tokens.len() < N_TOKENS {
        let pairs = generate_pairs(&words);
        let pair = find_top_pair(&pairs);
        update_words_with_new_token(&mut words, &pair);
        tokens.insert(pair.to_token());
    }
    tokens.insert("<unk>".to_string());
    tokens.insert(" ".to_string());

    println!("words: {:?}\n", words);
    println!("tokens: {:?}\n", tokens);

    let token_to_id: TokenToId = tokens.iter()
        .enumerate()
        .map(|(i, token)| (token.to_string(), i))
        .collect();
    println!("token_to_id: {:?}\n", token_to_id);

    let id_to_token: IdToToken = token_to_id.iter()
        .map(|(token, id)| (*id, token.clone()))
        .collect();
    let mut sorted_id_to_token: Vec<_> = id_to_token.iter().collect();
    sorted_id_to_token.sort_by_key(|(id, _)| *id);
    println!("id_to_token: {:?}\n", sorted_id_to_token);

    println!("word 'the' encoded: {:?}\n", encode_word(&token_to_id, "the"));
    println!("word 'cat' encoded: {:?}\n", encode_word(&token_to_id, "cat"));

    let sentence = "The cat found the hat.";
    println!("sentence '{}' encoded: {:?}\n", sentence, encode(&token_to_id, sentence));

    println!("sentence '{}' encoded and decoded: {}\n", sentence, decode(&id_to_token, &encode(&token_to_id, sentence)));
    let sentence = "The zebra lost the hat.";
    println!("sentence '{}' encoded and decoded: {}\n", sentence, decode(&id_to_token, &encode(&token_to_id, sentence)));
}