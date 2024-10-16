use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, Write};

/// Main function: Initializes QA data, computes TF-IDF, and runs the interactive question-answering loop
fn main() {
    let qa_data = initialize_qa_data().unwrap();
    let (tfidf_vectors, idf) = compute_tfidf(&qa_data);

    println!("Welcome to the Thoughtful AI Customer Support Agent!");
    println!("Ask a question about Thoughtful AI (type 'exit' to quit):");

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.eq_ignore_ascii_case("exit") {
            println!("Thank you for using the Thoughtful AI Customer Support Agent. Goodbye!");
            break;
        }

        let response = get_response(&qa_data, &tfidf_vectors, &idf, input);
        println!("{}", response);
    }
}

/// Initializes QA data by reading from a JSON file and parsing it into a HashMap
fn initialize_qa_data() -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    // Open the file
    let file = File::open("qa_data.json")?;
    let reader = BufReader::new(file);

    // Parse the JSON
    let json: Value = serde_json::from_reader(reader)?;

    // Create a HashMap to store the Q&A pairs
    let mut qa_data = HashMap::new();

    // Extract the questions and answers from the JSON
    if let Some(questions) = json["questions"].as_array() {
        for q in questions {
            if let (Some(question), Some(answer)) = (q["question"].as_str(), q["answer"].as_str()) {
                qa_data.insert(question.to_string(), answer.to_string());
            }
        }
    }

    Ok(qa_data)
}

/// Computes TF-IDF vectors for all questions in the QA data
/// Returns a tuple containing:
/// 1. A HashMap of TF-IDF vectors for each question
/// 2. The IDF (Inverse Document Frequency) scores for all words
fn compute_tfidf(
    qa_data: &HashMap<String, String>,
) -> (HashMap<String, HashMap<String, f64>>, HashMap<String, f64>) {
    let mut word_doc_count: HashMap<String, usize> = HashMap::new();
    let mut tfidf_vectors: HashMap<String, HashMap<String, f64>> = HashMap::new();

    // Compute document frequency
    for question in qa_data.keys() {
        let words: HashSet<String> = question
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();
        for word in words {
            *word_doc_count.entry(word).or_insert(0) += 1;
        }
    }

    // Compute IDF
    let doc_count = qa_data.len() as f64;
    let idf: HashMap<String, f64> = word_doc_count
        .iter()
        .map(|(word, count)| (word.clone(), (doc_count / *count as f64).ln()))
        .collect();

    // Compute TF-IDF
    for (question, _) in qa_data {
        let mut tf: HashMap<String, usize> = HashMap::new();
        let words: Vec<String> = question
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();
        for word in &words {
            *tf.entry(word.clone()).or_insert(0) += 1;
        }

        let mut tfidf = HashMap::new();
        for (word, count) in tf {
            let tf = count as f64 / words.len() as f64;
            let idf_value = idf.get(&word).unwrap_or(&0.0);
            tfidf.insert(word, tf * idf_value);
        }
        tfidf_vectors.insert(question.clone(), tfidf);
    }

    (tfidf_vectors, idf)
}

/// Finds the best matching question for the given input and returns the corresponding answer
fn get_response(
    qa_data: &HashMap<String, String>,
    tfidf_vectors: &HashMap<String, HashMap<String, f64>>,
    idf: &HashMap<String, f64>,
    input: &str,
) -> String {
    let input_vector = compute_input_vector(input, idf);
    let mut best_match = String::new();
    let mut max_similarity = f64::MIN;

    for (question, vector) in tfidf_vectors {
        let similarity = cosine_similarity(&input_vector, vector);
        if similarity > max_similarity {
            max_similarity = similarity;
            best_match = question.clone();
        }
    }

    if max_similarity > 0.5 {
        qa_data.get(&best_match).unwrap().clone()
    } else {
        if best_match.is_empty() {
            return "I'm sorry, I couldn't find a relevant question. Please try rephrasing your question.".to_string();
        } else {
            format!("I'm sorry, I don't have specific information about that. The closest question I can answer is: '{}'. Would you like me to answer that instead?", best_match)
        }
    }
}

/// Computes the TF-IDF vector for the input question
fn compute_input_vector(input: &str, idf: &HashMap<String, f64>) -> HashMap<String, f64> {
    let words: Vec<String> = input
        .to_lowercase()
        .split_whitespace()
        .map(String::from)
        .collect();
    let mut tf: HashMap<String, usize> = HashMap::new();
    for word in &words {
        *tf.entry(word.clone()).or_insert(0) += 1;
    }

    let mut tfidf = HashMap::new();
    for (word, count) in tf {
        let tf = count as f64 / words.len() as f64;
        let idf_value = idf.get(&word).unwrap_or(&0.0);
        tfidf.insert(word, tf * idf_value);
    }
    tfidf
}

/// Calculates the cosine similarity between two TF-IDF vectors
fn cosine_similarity(v1: &HashMap<String, f64>, v2: &HashMap<String, f64>) -> f64 {
    let mut dot_product = 0.0;
    let mut mag1 = 0.0;
    let mut mag2 = 0.0;

    for (word, value) in v1 {
        dot_product += value * v2.get(word).unwrap_or(&0.0);
        mag1 += value * value;
    }

    for value in v2.values() {
        mag2 += value * value;
    }

    dot_product / (mag1.sqrt() * mag2.sqrt())
}
