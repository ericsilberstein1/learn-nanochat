use pyo3::prelude::*;
use fancy_regex::Regex;
use rayon::prelude::*;
use std::collections::HashMap;
use ahash::AHashMap; // much faster
use ahash::AHashSet;
use compact_str::CompactString; // stores short strings on stack
use dary_heap::OctonaryHeap;
use std::cmp::Ordering;


type Pair = (u32, u32);

#[derive(Clone, Debug)]
struct Word {
    ids: Vec<u32>,
}

impl Word {

    #[inline]
    fn new(ids: Vec<u32>) -> Self {
        Self {
            ids: ids,
        }
    }

    #[inline]
    fn pairs<'a>(&'a self) -> impl Iterator<Item = Pair> + 'a {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    // merge all non-overlapping occurences of pair to new_id
    // returns a small Vec of local pair-count deltas for THIS word only <-- don't yet get what that means
    // 
    // so if we have a word like [1,2,3,1,2] and the new pair is (1,2) with id 4 I would expect the word
    // to get updated to: [4,3,4] but what would that return vec be? Maybe:
    //  [
    //     ((1,2), -2),
    //     ((2,3), -1)
    //     ((4,3),  1),
    //     ((3,1), -1)
    //     ((3,4),  1),
    //  ]
    // I guess that's the info we'll need to update our overall counts in an efficient way
    // wow there's a lot of bookeeping 
    // hmm, maybe it doesn't do -2, it just will list them as -1 twice
    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {

        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 {
            return Vec::new();
        }

        let mut out: Vec<u32> = Vec::with_capacity(n);
        let mut deltas: Vec<(Pair, i32)> = Vec::with_capacity(6);

        let mut i = 0;
        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i+1] == b {
                let left = out.last().copied(); // the id just to this left of the pair we're inserting
                let right = if i + 2 < n { Some(self.ids[i + 2]) } else {None}; // the id just to the right of the pair we're inserting
                
                if let Some(x) = left {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1)); // why not deltas.push((pair, -1)) ? 
                if let Some(y) = right {
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }

                out.push(new_id);
                i += 2;
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }

        self.ids = out;
        deltas
    }

}

#[derive(Debug, Eq, Clone)] // hmm do we really want pos sets to be compared to determine equality? maybe it's on only the ref? Ah this has to do with eq vs partialeq
struct MergeJob {
    pair: Pair,
    count: u64,
    pos: AHashSet<usize>,
}

impl PartialEq for MergeJob {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            other.pair.cmp(&self.pair)
        }
    }
}

#[pyclass]
pub struct Tokenizer {
    pub debug_print: bool, // extremely verbose, only use when training on tiny amounts of text
    pub merges: HashMap<Pair, u32>,
}

// why &[Word] vs &Vec<Word] ? the &[Word] is a slice, will we actually pass slices or always the whole vector?
// what does this do? say input is:
// words: [Word { ids: [32, 99, 97, 116] }, Word { ids: [116, 104, 101] }, Word { ids: [32, 116, 104, 101] }]
// counts: [2, 1, 1]
// the first return hashmap goes from pairs to counts, so maybe something like:
//     32, 99 -> 1
//     99, 97 -> 1
//     97, 116 -> 1
//     116, 104 -> 2
//     104, 101 -> 2
//     32, 116 -> 1
// but what is the second hashmap? maybe a list of the indices the pairs came from, like:
//     32, 99 -> 0
//     99, 97 -> 0
//     97, 116 -> 0
//     116, 104 -> 1,2
//     104, 101 -> 1,2
//     32, 116 -> 2
// makes sense so you don't have to go through all like I did in challenge 1  and 3 to find where to merge
// also wonder if the caller of this will merge just one pair. Maybe it could do multiple pairs like all with
// the same highest count or the n with the highest counts, but that also gets tricky
#[inline]
fn count_pairs_parallel(words: &[Word], counts: &[i32]) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {

    // hmm immediately see why it would be nice to iterate through a Word and get pairs and remember he built
    // something like that into Word, so do that first

    words.par_iter().enumerate().map(|(index, word)| {
            let mut pair_to_counts: AHashMap<Pair, i32> = AHashMap::new();
            let mut pair_to_indices: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            for pair in word.pairs() {
                *pair_to_counts.entry(pair).or_default() += counts[index];
                pair_to_indices.entry(pair).or_default().insert(index);
            }
            (pair_to_counts, pair_to_indices)
        })
        .reduce(|| (AHashMap::new(), AHashMap::new()),
                |(mut a_pair_to_counts, mut a_pair_to_indices), (b_pair_to_counts, b_pair_to_indices)| {
                    for (key, value) in b_pair_to_counts {
                        *a_pair_to_counts.entry(key).or_default() += value;
                    }
                    for (key, value) in b_pair_to_indices {
                        a_pair_to_indices.entry(key).or_default().extend(value);
                    }
                    (a_pair_to_counts, a_pair_to_indices)
                })
}

impl Tokenizer {

    // believe this will be like my main loop in challenges 1 and 3.
    // if the only text we fed in was "the cat the cat" then our input vectors will look
    // like: (keep in mind ' the' and 'the' are different)
    // words: [Word { ids: [32, 99, 97, 116] }, Word { ids: [116, 104, 101] }, Word { ids: [32, 116, 104, 101] }]
    // counts: [2, 1, 1]
    // 'cat', 'the', ' the'
    fn train_core_incremental(&mut self, mut words: Vec<Word>, counts: Vec<i32>, vocab_size: u32) {
        
        let num_merges = vocab_size - 256;

        // not yet sure why merges needs to be a field (instance variable), maybe even though we clear it, something about 
        // how this algoirthm works will cause it to end up with the ids of all tokens? Or maybe we only clear it this once
        // it's here in case train_core_incremental is called more than once?
        self.merges.clear();


        if self.debug_print {
            println!("Will now form and count all pairs, keeping track of which word(s) each pair comes from\n");
        }
        
        let (mut pair_counts, mut where_to_update) = count_pairs_parallel(&words, &counts);

        if self.debug_print {
            println!("pair -> count: {:?}\n", pair_counts);
            println!("pair -> indices: {:?}\n", where_to_update);
        }

        // next he builds a heap, why? Maybe there is a standard efficient way to do this that avoids
        // counting pairs from scratch over and over? maybe it's because we're going to want to find top-N?
        // ah, maybe we don't need to go from scratch because the count for pairs already found will never change, only 
        // pairs with new tokens are possible! so we only need to keep updating the heap with the latest to know what 
        // to merge on next
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, pos) in where_to_update.drain() {
            let c = *pair_counts.get(&pair).unwrap_or(&0);
            if c > 0 {
                heap.push(MergeJob { // does MergeJob implement something so the heap knows to order by count?
                    pair,
                    count: c as u64,
                    pos,
                });
            }
        }

        if self.debug_print {
            let sorted = heap.clone().into_sorted_vec();
            println!("Heap after adding initial pairs, shown sorted: {:?}\n", sorted);
        }

        let mut merges_done:u32 = 0;

        if self.debug_print {
            println!(
                "Will now merge until we have {} merges for a total vocab size of {} or there is nothing left to merge.\n",
                num_merges,
                vocab_size);
        }

        while merges_done < num_merges {
            let Some(mut top) = heap.pop() else { break };

            let current = *pair_counts.get(&top.pair).unwrap_or(&0);
            if top.count != current as u64 { // how does it get out of sync? prob will see that below
                top.count = current as u64;
                if top.count > 0 {
                    heap.push(top);
                }
                continue;
            }
            if top.count == 0 {
                break;
            }

            let new_id = 256 + merges_done;
            self.merges.insert(top.pair, new_id);

            if self.debug_print {
                println!("----- start of merge ------");
                println!("This merge is for {:?} with new id {}\n", top.pair, new_id);
            }

            // merge this pair in all words where it occurs
            let mut local_pos_updates: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            for &word_idx in &top.pos {
                if self.debug_print {
                    println!("About to merge pair {:?} into word {:?} with new id {:?}", top.pair, words[word_idx], new_id);
                }
                let changes = words[word_idx].merge_pair(top.pair, new_id);
                if self.debug_print {
                    println!("After merge, word is {:?}", words[word_idx]);
                    println!("and changes/deltas vector is {:?}", changes);
                }

                for (pair, delta) in changes {
                    let delta_total = delta * counts[word_idx];
                    if delta_total != 0 { // why would it ever be zero? can counts[word_idx] be 0?
                        *pair_counts.entry(pair).or_default() += delta_total;
                        if self.debug_print {
                            println!("changing global count for pair {:?} by {} to {}", pair, delta_total, pair_counts[&pair]);
                        }
                        if delta > 0 {
                            local_pos_updates.entry(pair).or_default().insert(word_idx);
                        }
                    }
                }

                if self.debug_print {
                    println!("");
                }
            }

            for (pair, pos) in local_pos_updates {
                let cnt = *pair_counts.get(&pair).unwrap_or(&0);
                if cnt > 0 {
                    // why isn't this also pushing pairs that are already in the heap causing duplicates?
                    // maybe because above, delta is only positive for pairs involving the new id introduced in this merge
                    // old pairs that were already in the heap can only go down in count 
                    // yes, suppose we have [1,2,3] and we merge in (2,3) with id 4
                    // (1,2) was already in the heap, but it goes down
                    // (1,4) goes up, but its brand new, so it couldn't have been in the heap
                    // This also explains why we need to do the lazy evaluation above, like in this example
                    // (1,2) will have the correct count pairs_count but a wrong (*higher*) count in the heap.
                    // It will get popped off "prematurely," the count will get corrected, and it will be pushed
                    // back on the heap. 
                    heap.push(MergeJob {
                        pair,
                        count: cnt as u64,
                        pos,
                    });
                }
            }

            merges_done += 1;

            if self.debug_print {
                println!("----- end of merge in of pair {:?} with id {} ------", top.pair, new_id);
                println!("state now:");
                println!("Words: {:?}\n", words);
                println!("Pair counts: {:?}\n", pair_counts);
                let sorted = heap.clone().into_sorted_vec();
                println!("Heap: {:?}\n", sorted);
                println!("Merges: {:?}", self.merges);
                println!("---------------------------------------------------\n");
            }
        }

        if self.debug_print {
            println!("Done merging! Total merges: {}", self.merges.len());
        }
    }
}

#[pymethods]
impl Tokenizer {

    #[new]
    #[pyo3(signature = (debug_print=false))]
    pub fn new(debug_print: bool) -> Self {
        Self {
            debug_print: debug_print,
            merges: HashMap::new(),
        }
    }

    pub fn train_from_iterator(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        pattern: String,
    ) -> PyResult<()> {

        let compiled_pattern = Regex::new(&pattern).unwrap();

        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);

        let py_iter: pyo3::Py<pyo3::PyAny> = unsafe {
            pyo3::Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
        };

        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            pyo3::Python::with_gil(|py| {
                buf.clear();
                let it = py_iter.bind(py);
                loop {
                    if buf.len() >= buffer_size {
                        return Ok(false);
                    }
                    let next_obj = unsafe {
                        pyo3::Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
                    };
                    match next_obj {
                        Some(obj) => {
                            buf.push(obj.extract()?);
                        }
                        None => {
                            if pyo3::PyErr::occurred(py) {
                                return Err(pyo3::PyErr::fetch(py));
                            } else {
                                return Ok(true);
                            }
                        }
                    }
                }
            })
        };

        let mut counts: AHashMap<CompactString, i32> = AHashMap::new();

        // under gil fill buf, outside of gil and in parallel split sentences into pieces and form a map from
        // piece to count (not sure why we don't go all the way down to byes and pairs of bytes at this point but
        // will see where this goes) -- I see then after he breaks each "chunk" into bytes

        if self.debug_print {
            println!("----start of iterating through text passed in, splitting it into words, and counting----\n");
        }

        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted {
                break;
            }
            if self.debug_print {
                println!("just filled one buffer under GIL, buffer contains {} strings", buf.len());
                println!("will now split into words and count in parallel without holding GIL");
            }

            // his code clones compiled_pattern, not sure why needed, leave out and see what breaks
            let local: AHashMap<CompactString, i32> = py.allow_threads(|| {
                buf.par_iter()
                    .map(|s| {
                        let mut m: AHashMap<CompactString, i32> = AHashMap::new();
                        for mat in compiled_pattern.find_iter(s) {
                            let piece = mat.unwrap().as_str();
                            *m.entry(CompactString::from(piece)).or_default() += 1;
                        }
                        m
                    })
                    .reduce(
                        || AHashMap::new(),
                        |mut a, b| {
                            for (key, value) in b {
                                *a.entry(key).or_default() += value;
                            }
                            a   
                        }
                    )
            });

            if self.debug_print {
                println!("finished splitting and counting for this buffer, 'local' counts map: {:?}\n", local);
            }

            for (key, value) in local {
                *counts.entry(key).or_default() += value;
            }

            if exhausted {
                break;
            }
        }

        if self.debug_print {
            println!("----end of iterating through text passed in, splitting it into words, and counting----\n");
            println!("counts map: {:?}\n", counts);
        }

        let mut words:Vec<Word> = Vec::with_capacity(counts.len());
        let mut cvec = Vec::with_capacity(counts.len());
        for (chunk, c) in counts.into_iter() {
            words.push(Word::new(chunk.as_bytes().iter().map(|&b| b as u32).collect()));
            cvec.push(c);
        }

        if self.debug_print {
            println!("words: {:?}\n", words);
            println!("cvec: {:?}\n", cvec);
            if words.len() > 0 {
                let pairs:Vec<Pair> = words[0].pairs().collect();
                println!("as an example, here are the pairs from the first word: {:?}\n", pairs);
            }
        }

        self.train_core_incremental(words, cvec, vocab_size);

        Ok(())
    }

    // this should return our tokens to ids, example
    // e   -> 101
    // h   -> 104
    // t   -> 116
    // th  -> 301
    // the -> 1015
    // We need to get this from merges, for example we could have:
    // (116, 104) -> 301
    // (301, 101) -> 1015
    // We could go through merges in id order and add to / look up in a new vector
    // 0 -> [0]
    // 1 -> [1]
    // ...
    // 255 -> [255]
    // ...
    // 301 -> [116] + [104] = [116, 104]
    // 1015 -> [116, 104] + [101] = [116, 104, 101]
    pub fn get_mergeable_ranks(&self) -> Vec<(Vec<u8>, u32)> {

        let mut ids: Vec<Vec<u8>> = Vec::with_capacity(256 + self.merges.len()); // [[0], [1], ..., [116, 104], ...]
        ids.extend( (0..=255).map(|i| vec![i]) );

        let mut merges: Vec<(&Pair, &u32)> = self.merges.iter().collect();
        merges.sort_by_key(|&(_, id)|  id);
        assert!(*merges[0].1 == 256);

        for ((a,b), id) in merges {
            let mut merged = ids[*a as usize].clone();
            merged.extend(&ids[*b as usize]);
            ids.push(merged);
            assert!(*id as usize == ids.len() - 1);
        }

        // go from [[0], [1], ... [116, 104]] to [([0],0), ([1],1), ..., ([116, 104], 301), ...]
        ids.into_iter().enumerate().map(|(i, merged)| (merged, i as u32)).collect()

        // above seems to work, now let's see how he did it
        // very similar!!
    }
}

// ---------------- "Play" ----------------
// Class to explore how rust works with python and try functions above, not actually part of tokenizing.

#[pyclass]
pub struct Play {

}

#[pymethods]
impl Play {

    #[new]
    pub fn new() -> Self {
        Self {}
    }

    pub fn hello(&self, name: String) -> String {
        format!("hello {}", name)
    }

    pub fn get_type(&self, obj: &pyo3::Bound<'_, pyo3::PyAny>) -> String {
        format!("{:?}", obj.get_type())
    }

    pub fn concat_from_iterator(&self, iterator: &pyo3::Bound<'_, pyo3::PyAny>) -> PyResult<String> {
        // this seems simple but the code for iterating in rustbpe is much more complicated,
        // maybe becuse it releases the GIL
        let mut result:String = String::new();
        for item in iterator.try_iter()? {
            result += item?.extract()?;
        }
        Ok(result)
    }

    pub fn fancy_concat_from_iterator(
        &self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>
    ) -> PyResult<String> {

        let mut result:String = String::new();

        let py_iter: pyo3::Py<pyo3::PyAny> = unsafe {
            pyo3::Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
        };

        let it = py_iter.bind(py);
        loop {
            let next_obj = unsafe {
                pyo3::Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
            };
            match next_obj {
                Some(obj) => {
                    result += obj.extract()?;
                }
                None => {
                    if pyo3::PyErr::occurred(py) {
                        return Err(pyo3::PyErr::fetch(py));
                    } else {
                        return Ok(result);
                    }
                }
            }
        }
    }

    pub fn find_matches(&self, pattern: String, text: String) -> Vec<String> {
        let compiled_pattern = Regex::new(&pattern).unwrap();
        compiled_pattern.find_iter(&text).map(|mat| mat.unwrap().as_str().to_string()).collect()
    }

    pub fn understand_comparison(&self) {
        let pair_a:Pair = (1,2);
        let pair_b:Pair = (3,4);
        let pair_c:Pair = (1,2);
        let pair_d:Pair = (1,3);
        println!("pair_a.cmp(pair_b): {:?}", pair_a.cmp(&pair_b));
        println!("pair_b.cmp(pair_a): {:?}", pair_b.cmp(&pair_a));
        println!("pair_a.cmp(pair_c): {:?}", pair_a.cmp(&pair_c));
        println!("pair_a.cmp(pair_d): {:?}", pair_a.cmp(&pair_d));
    }

    pub fn merge_pair_into_word(&self, word_ids: Vec<u32>, pair: Pair, new_id: u32) {
        let mut word = Word::new(word_ids);
        let deltas = word.merge_pair(pair, new_id);
        println!("word after merge: {:?}", word);
        println!("detals: {:?}", deltas);
    }

}

// ---------------- end "Play" ----------------

#[pymodule]
fn rust_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Play>()?;
    m.add_class::<Tokenizer>()?;
    Ok(())
}
