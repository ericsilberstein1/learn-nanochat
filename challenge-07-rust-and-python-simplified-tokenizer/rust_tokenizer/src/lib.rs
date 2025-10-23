use pyo3::prelude::*;
use fancy_regex::Regex;
use rayon::prelude::*;
use ahash::AHashMap; // much faster
use compact_str::CompactString; // stores short strings on stack

// ---------------- "Play" ----------------
// Classes and functions to explore how rust works with python, not actually part of tokenizing.

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

}

// ---------------- end "Play" ----------------


#[pyclass]
pub struct Tokenizer {
    pub debug_print: bool, // only use when training on tiny amounts of text
}

#[pymethods]
impl Tokenizer {

    #[new]
    pub fn new(debug_print: bool) -> Self {
        Self {
            debug_print: debug_print
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

        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted {
                break;
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
                println!("bottom of loop, 'local' counts map: {:?}", local);
            }

            for (key, value) in local {
                *counts.entry(key).or_default() += value;
            }

            if self.debug_print {
                println!("bottom of loop, length of 'global' counts map: {:?}\n", counts.len());
            }

            if exhausted {
                break;
            }
        }

        if self.debug_print {
            println!("Done iterating");
            println!("counts map: {:?}", counts);
        }

        let mut words:Vec<Vec<u32>> = Vec::with_capacity(counts.len()); // for now, will add Word type later
        let mut cvec = Vec::with_capacity(counts.len());
        for (chunk, c) in counts.into_iter() {
            words.push(chunk.as_bytes().iter().map(|&b| b as u32).collect());
            cvec.push(c);
        }

        if self.debug_print {
            println!("words: {:?}", words);
            println!("cvec: {:?}", cvec);
        }

        Ok(())
    }
}

#[pymodule]
fn rust_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Play>()?;
    m.add_class::<Tokenizer>()?;
    Ok(())
}
