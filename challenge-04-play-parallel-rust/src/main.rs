use std::time::Instant;
use rayon::prelude::*;

fn main() {

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
