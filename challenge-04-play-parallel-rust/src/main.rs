use std::time::Instant;
use rayon::prelude::*;
use std::collections::HashMap;

fn add_numbers() {
    let numbers: Vec<u64> = (1..1_000_000_000).collect();

    println!("--adding up {} numbers with regular iteration and fold--", numbers.len());
    let start = Instant::now();
    let sum = numbers.iter().fold(0, |acc, number| acc + number);
    println!("sum: {}", sum);
    println!("time elapsed: {:?}\n", start.elapsed());

    println!("--adding up {} numbers with parallel iteration and fold--", numbers.len());
    let start = Instant::now();
    let interim: Vec<u64> = numbers.par_iter().fold(|| 0, |acc, number| acc + number).collect();
    println!("interim.len(): {:?}", interim.len());
    println!("interim.sum(): {}", interim.iter().sum::<u64>());
    println!("time elapsed: {:?}\n", start.elapsed());
}

fn count_words() {
    let words:Vec<&str> = vec!["cat", "hat", "bat", "hat"]
        .iter()
        .cycle()
        .take(100_000_000)
        .copied()
        .collect();
    
    // for loop
    println!("--counting {} words with a for loop--", words.len());
    let start = Instant::now();
    let mut counts: HashMap<String, u32> = HashMap::new();
    for word in &words {
        *counts.entry(word.to_string()).or_insert(0) += 1;
    }
    println!("counts: {:?}", counts);
    println!("time elapsed: {:?}\n", start.elapsed());

    // for_each
    println!("--counting {} words with for_each--", words.len());
    let start = Instant::now();
    let mut counts: HashMap<String, u32> = HashMap::new();
    words.iter().for_each(|word| *counts.entry(word.to_string()).or_insert(0) += 1);
    println!("counts: {:?}", counts);
    println!("time elapsed: {:?}\n", start.elapsed());

    // parallel for_each
    // do we need to worry about locking?
    // yes! the compiler won't compile this code:
    // let mut counts: HashMap<String, u32> = HashMap::new();
    // words.par_iter().for_each(|word| *counts.entry(word.to_string()).or_insert(0) += 1);
    // println!("counts: {:?}", counts);
    // is there a thread safe HashMap?

    // parallel using fold and reduce
    println!("--counting {} words with parallel fold and reduce--", words.len());
    let start = Instant::now();
    let counts: HashMap<String, u32> = words
        .par_iter()
        .fold(|| HashMap::<String, u32>::new(),
            |mut counts, word| {
                *counts.entry(word.to_string()).or_insert(0) += 1;
                counts
            }
        )
        .reduce(
            || HashMap::<String, u32>::new(),
            |mut counts1, counts2| {
                for (key, value) in counts2 {
                    *counts1.entry(key).or_insert(0) += value;
                }
                counts1
            }
        );
    println!("counts: {:?}", counts);
    println!("time elapsed: {:?}\n", start.elapsed());
}

fn main() {

    add_numbers();
    count_words();

}