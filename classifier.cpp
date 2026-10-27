#include <iostream>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <cmath>
#include "csvstream.hpp"

using namespace std;

class Classifier
{
private:
  int total_posts = 0;
  int total_labels = 0;
  int vocab_size = 0;
  map<string, int> word_frequency;
  map<string, int> topic_frequency;
  map<string, map<string, int>> label_word_count;

public:
  // Reads training data from CSV file
  void train(csvstream &in, bool print)
  {
    if(!print){
      cout << "training data:" << endl;
    }

    map<string, string> row;
    while (in >> row) {
        string label = row["tag"];
        string content = row["content"];

        if(!print){
          cout << "  label = " << label << ", content = " << content << endl;
        }

        set<string> words = unique_words(content);

        //Update Member variables
        total_posts++;
        if(topic_frequency.find(label) == topic_frequency.end()){
          total_labels++;
        }
        topic_frequency[label] += 1;

        for(const auto &w : words){
          word_frequency[w] += 1;
          label_word_count[label][w] += 1;
        }
    }
    vocab_size = word_frequency.size();
    
  }

  //Reads data from test file
  void read(csvstream &test){

    cout << "\ntest data:" << endl;

    double highest = -1e308;
    string highest_label;
    double probability;
    map<string, string> row;
    int correct_count = 0;
    int count = 0;

    while (test >> row) {
        string correct_tag = row["tag"];
        string content = row["content"];

        cout << "  correct = " << correct_tag << ", predicted = ";

        set<string> words = unique_words(content);
        
        for(const auto &key : topic_frequency){
          string tag = key.first;
          probability = log_prior(tag);
          for(const auto &s : words){
            probability += log_likelihood(tag, s);
          }

          if(probability > highest){
            highest = probability;
            highest_label = tag;
          }
        }

        cout << highest_label << ", log-probability score = " << highest << endl;
        cout << "  content = " << content << endl << endl; 

        count++;
        if(correct_tag == highest_label){
          correct_count++;
        }

        highest = -1e308;
        highest_label = "";
    }

    cout << "performance: " << correct_count << " / " << count 
      << " posts predicted correctly" << endl;
  }

  double log_prior(const string &label)
  {
    return log(topic_frequency[label] / static_cast<double>(total_posts));
  }

  double log_likelihood(const string &label, const string &word){
    double answer = 0;

    //Case 1: word seen with this label
    if (label_word_count[label].count(word) > 0) {
        double denominator = static_cast<double>(topic_frequency[label]);
        answer = log(label_word_count[label][word] / denominator);
    }
    //Case 2: word exists in training data but not with this label 
    else if (word_frequency.count(word) > 0) {
        answer = log(word_frequency[word] / static_cast<double>(total_posts));
    } 
    //Case 3: word never seen in training data at all
    else {
        answer = log(1 / static_cast<double>(total_posts));
    }
    
    return answer;
  }

  // EFFECTS: Return a set of unique whitespace delimited words
  set<string> unique_words(const string &str)
  {
    istringstream source(str);
    set<string> words;
    string word;
    while (source >> word)
    {
      words.insert(word);
    }
    return words;
  }

  int get_total_posts(){
    return total_posts;
  }

  void print_training_data(){
    cout << "trained on " << get_total_posts() << " examples" << endl;
    cout << "vocabulary size = " << vocab_size << endl << endl;

    cout << "classes:" << endl;
    for(const auto &s : topic_frequency){
      cout << "  " << s.first << ", " << s.second << " examples, log-prior = "
        << log_prior(s.first) << endl;
    }

    cout << "classifier parameters:" << endl;
    for(const auto &label : label_word_count){
      for(const auto &word : label_word_count[label.first]){
        cout << "  " << label.first << ":" << word.first << ", count = " << word.second
          << ", log-likelihood = " << log_likelihood(label.first, word.first) << endl;
      }
    }

  }
};

int main(int argc, char *argv[])
{
  cout.precision(3);

  if (argc != 2 && argc != 3) {
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
    return 1;
  }

  csvstream training_file(argv[1]);

  bool print = false;
  if(argc == 3){
    print = true; 
  }

  Classifier c;
  c.train(training_file, print);

  if(argc == 2){
    c.print_training_data();
  }
  else{
    cout << "trained on " << c.get_total_posts() << " examples" << endl;
  }
  
  if (argc == 3) {
    csvstream test_file(argv[2]);
    c.read(test_file);
  }
}