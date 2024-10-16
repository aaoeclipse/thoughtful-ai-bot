# AI Bot Helper - AI Technical Screen

## Summary

Hello Thoughtful AI! For this project I've created a simple AI, taking into account the amount of time required to program and the hard coded responses I felt that it could be done not by machine learning but instead using a more mathematical/algorithmical approach. Here are the reasons for my decisions on this project.

1. **Rust**: Rust is a low-level high fidelity programing langauges that will not only let us deploy the bot in a smaller server but since it's main focus is security we can be much more confident that it will not behave unexpectedly and keep us secure from user input. It would also be very fast, much faster than high level langagues like js and python making it able to handle more requests per seconds at a cheaper price.

2. **NonML Approach/TF-IDF**: Term Frequency-Inverse Document Frequency is used a lot for information retrieval, text mining, and NLP. It tries to calculate the impratance of a word in a text, we can do this for the input of the user and the hardcoded QA to get a vector on both of them and compare them either by using euclidian distance, or in this case, cosine similarity. It's a fast algorithm O(n) and since it's not ML it doesn't need any traning. One could argue that maybe word embedding could produce better results however it is more computationally intensive, and needs traning. If we are to deploy the bot we should take into account that processing means money and we would need to get a better idea of how much the company is willing to pay for the program to be more accurate. 

3. **JSON file instead of hardcoding**: Having it be read from a file help with the flexibility to be able to just modify the file without modifying the source code. 

4. **Logic**: I also added some custom text for when the program is not sure what to pick, it will display to the user that it didn't understand but that it can answer another question, and proceeds to show the most likly.

## Running the program
The executables are in the folder executables/ I've compiled it for Linux and windows. You can run any of them as they were a simple executable.

## Setup for Rust
1. Install rust
2. Install dependencies and build
```sh
cargo build 
```
3. Run
```sh
cargo run
```