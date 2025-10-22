use std::collections::HashSet;
use std::collections::HashMap;

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

fn generate_pairs(words: &Vec<Vec<String>>) -> HashMap<Pair, u32> {
    let mut pairs: HashMap<Pair, u32> = HashMap::new();
    
    for word in words {
        for i in 0..word.len()-1 {
            let pair = Pair{first: word[i].to_string(), second: word[i+1].to_string()};
            let count = pairs.entry(pair).or_insert(0);
            *count += 1;
        }
    }
    return pairs;
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
    return candidate_pair.unwrap();
}

fn update_words_with_new_token(words: &mut Vec<Vec<String>>, new_pair: &Pair, new_token: &str) {
    for word in words.iter_mut() {
        let mut i: usize = 0;
        while i < word.len() - 1 {
            if (Pair{first: word[i].clone(), second: word[i+1].clone()}) == *new_pair { // ??? why do I need that clone?
                let mut new_word = vec![];
                new_word.extend_from_slice(&word[..i]);
                new_word.push(new_token.to_string());
                new_word.extend_from_slice(&word[i+2..]);
                *word = new_word;
            }
            i += 1
        }
    }
}

const N_TOKENS: usize = 20;

fn encode_word(token_to_id: &HashMap<&String, usize>, word: &str) -> Vec<usize> {
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
            token_id = token_to_id.get(&"<unk>".to_string()).copied();
        }
        encoded.push(token_id.unwrap());
        unencoded_part = unencoded_part.chars().skip(i+1).collect();
    }
    return encoded;
}

fn encode(token_to_id: &HashMap<&String, usize>, sentence: &str) -> Vec<usize> {
    let mut encoded: Vec<usize> = vec![];
    for word in sentence.split(' ') {
        encoded.extend(encode_word(token_to_id, word));
        encoded.push(token_to_id[&" ".to_string()]);
    }
    encoded.pop();
    return encoded;
}

fn decode(id_to_token: &HashMap<usize, &String>, encoded_sentence: &Vec<usize>) -> String {
    return encoded_sentence.iter().map(|id| id_to_token[id].as_str()).collect();
}

fn main() {
    let corpus = "The batat and the cat fought over the hat.";

    let mut words: Vec<Vec<String>> = vec![];
    for word in corpus.split(' ') {
        words.push(word.chars().map(|c| c.to_string()).collect());
    }

    let mut tokens: HashSet<String> = HashSet::new();

    for word in &words {
        for c in word {
            tokens.insert(c.to_string());
        }
    }

    while tokens.len() < N_TOKENS {
        let pairs = generate_pairs(&words);
        let pair = find_top_pair(&pairs);
        update_words_with_new_token(&mut words, &pair, &pair.to_token());
        tokens.insert(pair.to_token());
    }
    tokens.insert("<unk>".to_string());
    tokens.insert(" ".to_string());

    println!("--words--");
    for word in &words {
        println!("------");
        for c in word {
            println!("{}", c);
        }
    }

    println!("--tokens--");
    for token in &tokens {
        println!("{}", token);
    }

    let token_to_id: HashMap<&String, usize> = tokens.iter() // should this be String not &String ?
        .enumerate()
        .map(|(i, token)| (token, i))
        .collect();

    let id_to_token: HashMap<usize, &String> = token_to_id.iter()
        .map(|(token, id)| (*id, *token))
        .collect();

    println!("token_to_id: {:?}", token_to_id);
    let mut sorted: Vec<_> = id_to_token.iter().collect();
    sorted.sort_by_key(|(id, _)| *id);
    println!("id_to_token: {:?}", sorted);

    let foo: Vec<usize> = encode_word(&token_to_id, "the");
    println!("foo is {:?}", foo);

    let bar = encode(&token_to_id, "The cat found the hat.");
    println!("bar is {:?}", bar);

    println!("it is {}", decode(&id_to_token, &bar));

    println!("{}", decode(&id_to_token, &encode(&token_to_id, "The cat found the hat.")));
    println!("{}", decode(&id_to_token, &encode(&token_to_id, "The zebra lost the hat.")));


}
