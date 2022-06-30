# Simon Says
In this repo we build a chat bot named Simon using a LSTM Neural Network. This is completed by leveraging Python, NLTK and Keras. This is a small example of how large scale and intricate chat bots could be made. This model is based off of a dataset called "intents" which contains the labeled data that is neccessary for trainings.

# How does it work (non-technical)?

### Starting with the data
The input data that we use to train the model is composed of three separate parts.
- tag - The type of question. For instance "See you later!" is categorized as a "goodbye" tag. 
- patterns - Examples of what we believe users might ask. For example "See you later!" is the pattern. This is used to train to model so that it can detect which category user's inputs belong to. 
- responses - Once the model knows what category the question relates to it picks randomly from predefined respones. A response to "See you later!" might be ""Have a nice day".
-  Example Data:
```
         {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
         "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."]
        }
```
### Cleaning your input sentence:
When you input a sentence, it is first cleaned before entering the model. Meaning all the superfluous grammatical structure of a sentence is wiped out. Words like "helping" become "help" and stop words  such as "and" or "but" are removed since they carry little meaning that indicates what the sentence is about. So the model only gets the bare bones of the sentence. This same cleaning is done on the original "patterns" used in training.

### Model
Some might expect that the model is generating the text that is used as a response, but that is not how this chatbot works. The model in this case classifies which response bucket "Simon" should choose from, given the user's inputted sentence. All the responses from Simon are predetermined by the developer. The model only determines which category your reponse belongs to based on what it has seen before in the "pattern" training data. Diving a bit deeper - the model looks for words that exist both in the inputed sentence and the training "patterns". The highest matched category or "tag" for the given input sentence is deemed the appropriate "tag". Once the tag is determined, Simon chooses a random response from the list of possible reponses in that "tag".

### Model Parameters
- LSTM neural network.
- 3 layers with 100 and 50 neurons in the first two layers with the last layer having neurons equal to the number of intents.
- SGD as optimizer 
- Accuracy as our objective function that we optimize against.
- 220 epochs with a batch size of 5.

# Running the App
- Clone repo
- Set up python virtual environment natively
  - python -m venv < location-of-project >\simon_says\venv
    - This creates a virtual environment and a folder in your project
  - Activate virtual environment
  
    Windows
    - < location-of-project >\simon_says\venv\Scripts\activate
    
    MAC
    - source < location-of-project >/venv/bin/activate 
- Set up python virtual environment with conda
  - conda create -n simonsays python=3.6
  - conda activate simonsays 
- Run: pip install -r requirements.txt
- Train the model (run train_simon.py) 
- Run the graphical interface ( run simons_gui)
- Input text to Simon and see his responses!
- Update the intents.json to start creating your own chatbot!

# Citation
Model was developed with data and inspiration from https://data-flair.training/blogs
