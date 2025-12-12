# CIS 405/505 – Algorithm Design Project  
## Text Summarization Using TF–IDF and TextRank

### Author: Michael Acquah  
### Course: CIS 405/505 – Algorithm Design  
### Semester: Fall 2025

---

## 1. Project Overview
This project implements two different extractive text summarization algorithms:

1. **TF–IDF (Term Frequency–Inverse Document Frequency)**
2. **TextRank (Graph-Based Ranking Algorithm)**

Both algorithms take a text file or input string, preprocess the sentences, compute sentence importance based on their respective scoring approaches, and output a summary consisting of the top *K* sentences.

The project was completed as part of the Algorithm Design course requirements, covering all steps from problem definition through algorithm design, analysis, implementation, testing, evaluation, and final reporting.

---

## 2. File Structure

project-folder/
│
├── textrank.py # TextRank algorithm implementation
├── tfidf.py # TF–IDF algorithm implementation
├── README.md # This file
└── Algorithm_project.pdf # Final report (steps 1–11)

yaml
Copy code

---

## 3. Requirements

### **Python Version**
- Python **3.8+** recommended
- No external libraries required beyond the standard library (re, math, collections)

### **Dependencies**
Both programs use only built-in Python modules:
- `re`
- `math`
- `collections.Counter`
- `collections.defaultdict`

No installation needed.

---

## 4. How to Run the Code

You may run either algorithm from the command line or inside VS Code / PyCharm.

### **4.1 Running TF–IDF**

```bash
python tfidf.py
 ```
---

### **4.2 Running TextRank**

```bash
python textrank.py
```

### **4.3 Using a Text File as Input**
You may replace the text variable with:
with open("sample_input.txt", "r") as f:
    text = f.read()
