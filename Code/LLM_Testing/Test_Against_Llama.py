from groq import Groq
import csv
import requests
import re
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import pandas as pd
import ast
import math
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
import ast


class Llama_Test:
    def __init__(self):
        # nltk.download('wordnet')
        # Stop words
        self.STOP_WORDS = set(
            """
        a about above across after afterwards again against all almost alone along
        already also although always am among amongst amount an and another any anyhow
        anyone anything anyway anywhere are around as at

        back be became because become becomes becoming been before beforehand behind
        being below beside besides between beyond both bottom but by

        call can cannot ca could

        did do does doing done down due during

        each eight either eleven else elsewhere empty enough even ever every
        everyone everything everywhere except

        few fifteen fifty first five for former formerly forty four from front full
        further

        get give go

        had has have he hence her here hereafter hereby herein hereupon hers herself
        him himself his how however hundred

        i if in indeed into is it its itself

        keep

        last latter latterly least less

        just

        made make many may me meanwhile might mine more moreover most mostly move much
        must my myself

        name namely neither never nevertheless next nine no nobody none noone nor not
        nothing now nowhere

        of off often on once one only onto or other others otherwise our ours ourselves
        out over own

        part per perhaps please put

        quite

        rather re really regarding

        same say see seem seemed seeming seems serious several she should show side
        since six sixty so some somehow someone something sometime sometimes somewhere
        still such

        take ten than that the their them themselves then thence there thereafter
        thereby therefore therein thereupon these they third this those though three
        through throughout thru thus to together too top toward towards twelve twenty
        two

        under until up unless upon us used using

        various very very via was we well were what whatever when whence whenever where
        whereafter whereas whereby wherein whereupon wherever whether which while
        whither who whoever whole whom whose why will with within without would

        yet you your yours yourself yourselves
        """.split()
        )

        self.contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
        self.STOP_WORDS.update(self.contractions)

        for apostrophe in ["‘", "’"]:
            for stopword in self.contractions:
                self.STOP_WORDS.add(stopword.replace("'", apostrophe))
        
        self.stops = list(list(set(self.STOP_WORDS)) + ["http"])



        # Setting up the emotion dictionary
        fileEmotion = r"C:\Users\natha\Programming\LLM_Research\LLM_Semantics\Code\emotion_itensity.txt" #change file path
        table = pd.read_csv(fileEmotion,  names=["word", "emotion", "itensity"], sep='\t')
        self.emotion_dic = dict()
        self.lmtzr = WordNetLemmatizer()
        for index, row in table.iterrows():
            #add first as it is given in the lexicon
            temp_key = row['word'] + '#' + row['emotion']
            self.emotion_dic[temp_key] = row['itensity']

            #add in the normal noun form
            temp_key_n = self.lmtzr.lemmatize(row['word']) + '#' + row['emotion']
            self.emotion_dic[temp_key_n] = row['itensity']
            
            #add in the normal verb form
            temp_key_v = self.lmtzr.lemmatize(row['word'], 'v') + '#' + row['emotion']
            self.emotion_dic[temp_key_v] = row['itensity'] 

    def Cleanup_Description(description):
        filtered = re.sub("[^a-zA-Z]+", " ", description) # replace all non-letters with a space
        pat = re.compile(r'[^a-zA-Z ]+')
        filtered = re.sub(pat, '', filtered).lower() #  convert to lowercase
        filtered = ' '.join(filtered.split())
        return filtered

    def Filter_Sample(file_path, lower_age, upper_age):
        """
        The purpose of this function is to take the data set data and vector data from "complete_pl_output.csv" and
        filter that down into the age group that we want to test with the LLM's. The function will output two files
        from the data: a Sample_Data.txt file with book descriptions that the LLM can read in and produce recommendations
        for, and a Filtered_Data.txt file which will have all the data for all the descriptions written to Sample_Data.txt.
        """
        with open(file_path, encoding="utf-8") as raw_data:
            with open("Sample_Data.txt", "w") as descriptions:
                with open("Filtered_Data.txt", "w") as filtered_raw:
                    csv_reader = csv.reader(raw_data)
                    writer_descriptions = csv.writer(descriptions)
                    writer_filtered_raw = csv.writer(filtered_raw)
                    writer_filtered_raw.writerow(next(csv_reader))
                    for line in csv_reader:
                        # 7 is the index number for the age in the file "complete_pl_output"
                        try:
                            if lower_age <= int(line[7]) <= upper_age:
                                # 9 is the index number for the description of the book
                                filtered_line = Llama_Test.Cleanup_Description(line[9])
                                writer_descriptions.writerow([filtered_line])
                                writer_filtered_raw.writerow(line)
                        except:
                            print("Error while filtering:\n" + line[9])


    def Recommend_for_Sample(sample_file, output_file):
        with open(sample_file) as inf:
            descriptions = inf.read()
        with open("API_Key.txt") as api_file:
            key = api_file.read()
        client = Groq(api_key=key)
        completion = client.chat.completions.create(
            # Options: llama-3.3-70b-versatile, mixtral-8x7b-32768, mistral-saba-24b, gemma2-9b-it
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    # Previous Description: "For all of the following list of book descriptions, without looking up the corresponding book or recommended age online, print 5 book recommendations followed by the given description on each line (of the format description:...description here...), separated by commas (ex: book_title_1, book_title_2,...book_title_5, description...and go on to the next line) for someone who read the book of the given description. Do this for ALL descriptions, even though the list is long. I'm writing this to a csv file. Output nothing else as it might mess up my program when I go in and read the csv file. Here are the descriptions: " + descriptions
                    # Second Previous Description: ""For each of the following book descriptions, without looking up the corresponding book or recommended age online, print a 6 column csv file with the first five columns being five book recommendations for another reader of the same age as the one who enjoyed that book, and the sixth column being the original book description. (Example of one line: Book Recommendation 1, Book Recommendation 2, Book Recommendation 3, Book Recommendation 4, Book Recommendation 5, Original Book description). Here are the descriptions: " + Descriptions
                    "content": "For each of the following 5 book descriptions, without looking up the corresponding book or recommended age online, I want you to print an 21 columns of a csv file (Which I will save to a CSV file) with the first 20 columns being 20 book title recommendations ranked by how good they are for another reader of the same age as the one who enjoyed that book, and the 21st column being the original book description. (Example of one line: Top Recommendation, Second Best Recommendation, 3rd Best Recommendation,..., 20th description, Original Book description). Do this for all book descriptions. Here are the descriptions: " + descriptions
                },
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        with open(output_file, 'w') as of:
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    of.write(chunk.choices[0].delta.content)

    def Grab_Published_Description(title):
        """
        The purpose of this program is to take a book title, search the internet, and grab the book's published description, so that we can run the emotion vector calculator on
        each of the recommendations that the LLM gives us for each sample description.
        """
            # Google Books API endpoint
        url = f"https://www.googleapis.com/books/v1/volumes?q={title}"

        # Sending GET request to the Google Books API
        response = requests.get(url)

        # If the response is successful (status code 200)
        if response.status_code == 200:
            data = response.json()
            
            # Check if any items are returned
            if "items" in data:
                book = data["items"][0]  # Taking the first result (you can handle multiple results if needed)
                
                # Extract description
                description = book["volumeInfo"].get("description", "No description available")
                
                return description
            else:
                return "No books found for that title."
        else:
            return f"Error: Unable to fetch data (Status code {response.status_code})"
        
    def Test_LLM_Recommendations(Recommendation_File):
        """
        The purpose of this function is to take in the LLM_Recommendations file, calculate the emotion vector for the sample description given,
        calculate the emotion vectors for each recommendation, and then take an average of these emotion vectors for final analysis.
        """
        pass

    def isStopWord(self, word):
        if word in self.stops:
            return True
        else:
            return False

    def isWordInEmotionFile(self, word):
        # Slightly faster implementation
        for key in self.emotion_dic.keys():
            if key.startswith(word + "#"):
                return True
        return False

    #function that get the emotion itensity
    def getEmotionItensity(self, word, emotion):
        key = word + "#" + emotion
        try:
            return self.emotion_dic[key]
        except:
            return 0.0

    #Assign the emotion itensity to the dictionary
    def calculateEmotion(self, emotions, word):
        emotions["Anger"] += self.getEmotionItensity(word, "anger")
        emotions["Anticipation"] += self.getEmotionItensity(word, "anticipation")
        emotions["Disgust"] += self.getEmotionItensity(word, "disgust")
        emotions["Fear"] += self.getEmotionItensity(word, "fear")
        emotions["Joy"] += self.getEmotionItensity(word, "joy")
        emotions["Sadness"] += self.getEmotionItensity(word, "sadness")
        emotions["Surprise"] += self.getEmotionItensity(word, "surprise")
        emotions["Trust"] += self.getEmotionItensity(word, "trust")

    #get the emotion vector of a given text TODO: This one we would need to change
    def getEmotionVector(self, text, removeObj = False, useSynset = True):
        #create the initial emotions
        emotions = {"Anger": 0.0,
                    "Anticipation": 0.0,
                    "Disgust": 0.0,
                    "Fear": 0.0,
                    "Joy": 0.0,
                    "Sadness": 0.0,
                    "Surprise": 0.0,
                    "Trust": 0.0,
                    "Objective": 0.0}
        #parse the description
        str = re.sub("[^a-zA-Z]+", " ", text) # replace all non-letters with a space
        pat = re.compile(r'[^a-zA-Z ]+')
        str = re.sub(pat, '', str).lower() #  convert to lowercase

        #split string
        splits = str.split()
        
        #iterate over words array
        for split in splits:
            if not self.isStopWord(split):
                #first check if the word appears as it does in the text
                if self.isWordInEmotionFile(split): 
                    self.calculateEmotion(emotions, split)
                    
                # check the word in noun form (bats -> bat)
                elif self.isWordInEmotionFile(self.lmtzr.lemmatize(split)):
                    self.calculateEmotion(emotions, self.lmtzr.lemmatize(split))
                    
                # check the word in verb form (ran/running -> run)
                elif self.isWordInEmotionFile(self.lmtzr.lemmatize(split, 'v')):
                    self.calculateEmotion(emotions, self.lmtzr.lemmatize(split, 'v'))  
                    
                # check synonyms of this word
                elif useSynset and wordnet.synsets(split) is not None:
                    # only check the first two "senses" of a word, so we don't stray too far from its intended meaning
                    for syn in wordnet.synsets(split)[0:1]:
                        for l in syn.lemmas():
                            if self.isWordInEmotionFile(l.name()):
                                self.calculateEmotion(emotions, l.name())
                                continue
                                
                    # none of the synonyms matched something in the file
                    emotions["Objective"] += 1
                    
                else:
                    # not found in the emotion file, assign a score to Objective instead
                    emotions["Objective"] += 1

        # remove the Objective category if requested
        if removeObj:
            del emotions['Objective']
            
        total = sum(emotions.values())
        for key in sorted(emotions.keys()):
            try:
                # normalize the emotion vector
                emotions[key] = (1.0 / total) * emotions[key]
            except:
                emotions[key] = 0

        return emotions
    
    def Convert_to_Vectors(input_file, output_csv):
        """
        The purpose of this function is to take in the LLM recommendations that we've got and to assign an emotion vector
        to each of the book recommendations given, along with the original description given as input. This is in hopes of
        being able to better analyze how well the LLM did in its recommendations.
        """
        nltk.download('wordnet')
        with open(input_file) as inf:
            with open(output_csv,'w', encoding='utf-8') as of:
                writer = csv.writer(of)
                writer.writerow(["Book 1","Book 2","Book 3","Book 4", "Book 5","Original Description"])
                lines = inf.readlines()
                for line in lines:
                    if line.strip():
                        inputs = line.split(',')
                        outputs = []
                        filler_object = Llama_Test()
                        books = inputs[:5]
                        original_input = inputs[-1]
                        for book in books:
                            book_description = Llama_Test.Grab_Published_Description(book.strip())
                            outputs.append(Llama_Test.getEmotionVector(filler_object, book_description))
                        outputs.append(Llama_Test.getEmotionVector(filler_object, original_input.lstrip("description: ")))
                        writer.writerow(outputs)
                    
    def Calculate_Vector_Distance(vector_1, vector_2):
        """
        The purpose of this program is to take in two dictionary emotion vectors and to calculate the distance between the two
        in order to determine how well the LLM makes its recommendations relative to the original description.
        """
        distance = 0
        for key in vector_1:
            distance += (vector_2[key] - vector_1[key])**2

        return math.sqrt(distance)


    def Determine_Data_Difference(vector_csv, output_file):
        """
        The purpose of this function is to take in a file of the calculated vectors, from both the LLM recommendations and the
        original description, and to analyze the difference between the vector average of the recommendations with the original
        description. This is in hopes of noting statistical significance between the difference and coming to a scientific
        conclusion.
        """
        with open(vector_csv) as inf:
            with open(output_file, 'w', encoding='utf-8') as of:
                reader = csv.reader(inf)
                next(reader)
                for row in reader:
                    if row:
                        vectors = [ast.literal_eval(entry) for entry in row]
                        average_recommendation_vector = vectors[0]
                        for vector in vectors[1:5]:
                            for key in vector:
                                average_recommendation_vector[key] += vector[key]

                        for key in average_recommendation_vector:
                            average_recommendation_vector[key] /= 5

                        distance = Llama_Test.Calculate_Vector_Distance(average_recommendation_vector, vectors[-1])
                        
                        of.write(f"Average Recommendation Vector: {average_recommendation_vector}")
                        of.write(f"Input Description vector: {vectors[-1]}\n")
                        of.write(f"Distance between emotion vectors: {distance}\n\n")

    def get_book_page(self, book_title, headers, base_url):
        search_url = f"{base_url}/search?utf8=%E2%9C%93&q={quote_plus(book_title)}"
        response = requests.get(search_url, headers=headers)
        if response.status_code != 200:
            raise Exception("Failed to fetch search page")
        
        soup = BeautifulSoup(response.text, "html.parser")
        first_result = soup.select_one(".bookTitle")  # Finds first book link
        if not first_result:
            raise Exception("No book results found")
        
        return base_url + first_result["href"]

    def get_similar_books_url(book_url):
        """Use Selenium to extract the similar books URL from the book's page."""
        # Set up the Service object with the ChromeDriver path
        service = Service(ChromeDriverManager().install())
        # Initialize the WebDriver with the service
        driver = webdriver.Chrome(service=service)
        try:
            driver.get(book_url)
            similar_link = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//a[@aria-label='Tap to show all similar books']"))
            )
            return similar_link.get_attribute('href')
        finally:
            driver.quit()

    def get_similar_books(self, similar_url, headers):
        response = requests.get(similar_url, headers=headers)
        if response.status_code != 200:
            raise Exception("Failed to fetch similar books page")
        
        soup = BeautifulSoup(response.text, "html.parser")
        book_elements = soup.select(".coverWrapper + a")[:10]  # Selects first 10 books
        
        return [book.get_text(strip=True) for book in book_elements]

    def get_recommended_books(self, book_title):
        headers = {"User-Agent": "Mozilla/5.0"}  # Define headers here
        base_url = "https://www.goodreads.com"  # Define base_url here
        book_url = self.get_book_page(book_title, headers, base_url)
        similar_url = self.get_similar_books_page(book_url, headers, base_url)
        return self.get_similar_books(similar_url, headers)

    def find_cosine_similarity(self, ev1, ev2):
        """Calculate the cosine similarity between two emotion vectors."""
        dot_product = sum(ev1[key] * ev2.get(key, 0) for key in ev1)
        magnitude1 = math.sqrt(sum(value ** 2 for value in ev1.values()))
        magnitude2 = math.sqrt(sum(value ** 2 for value in ev2.values()))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def find_top_similar_books(self, input_vector, target_age, file_path=r"C:\Users\annie\.vscode\LLM_Semantics\Data\Teenager_GoodReads_Emotion.csv"):
        """
        Find the top 20 books with emotion vectors most similar to the input vector,
        for entries where the age is within 1 year of the target age.

        Args:
            input_vector (dict): Input emotion vector with ~10 dimensions (e.g., {'happy': 0.5, 'sad': 0.3, ...}).
            target_age (float): Target age to filter entries (e.g., 16.5).
            file_path (str): Path to the CSV file containing book data.

        Returns:
            dict: Dictionary mapping book titles (str) to their cosine similarity scores (float)
                for the top 20 most similar books. Returns fewer than 20 if not enough entries match.
                Returns empty dict if no entries match or file cannot be processed.
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, header=None, quotechar='"', escapechar=None)

            # Assign column names based on the specified structure
            df.columns = ['ISBN','ID','Title','Author','Description','Average_Rating','Age','EmotionVector']

            # Initialize lists to store filtered data
            book_titles = []
            similarities = []
            seen_titles = set()  # To track duplicates

            # Process each row individually to handle errors and duplicates
            for idx, row in df.iterrows():
                try:
                    # Extract and validate age
                    age = pd.to_numeric(row['Age'], errors='coerce')
                    if pd.isna(age):
                        print(f"Error at line {idx + 2}: Invalid age value '{row['Age']}'. Skipping this line.")
                        continue

                    # Check if age is within 1 year of target_age
                    if not (target_age - 1 <= age <= target_age + 1):
                        continue

                    # Extract book title
                    book_title = row['Title']
                    if pd.isna(book_title) or not isinstance(book_title, str) or not book_title.strip():
                        print(f"Error at line {idx + 2}: Invalid book title '{book_title}'. Skipping this line.")
                        continue

                    # Check for duplicates
                    if book_title in seen_titles:
                        print(f"Duplicate book title '{book_title}' found at line {idx + 2}. Skipping this line.")
                        continue
                    seen_titles.add(book_title)

                    # Parse emotion vector
                    emotion_vector_str = row['EmotionVector']
                    try:
                        emotion_dict = ast.literal_eval(emotion_vector_str)
                        if not isinstance(emotion_dict, dict) or not emotion_dict:
                            print(f"Error at line {idx + 2}: Emotion vector '{emotion_vector_str}' is not a valid dictionary. Skipping this line.")
                            continue
                    except (ValueError, SyntaxError) as e:
                        print(f"Error at line {idx + 2}: Failed to parse emotion vector '{emotion_vector_str}'. {str(e)}. Skipping this line.")
                        continue

                    # Calculate cosine similarity
                    similarity = self.find_cosine_similarity(input_vector, emotion_dict)

                    # Store the result
                    book_titles.append(book_title)
                    similarities.append(similarity)

                except Exception as e:
                    print(f"Error at line {idx + 2}: Unexpected error processing row: {str(e)}. Skipping this line.")
                    continue

            if not book_titles:
                print("No valid entries found within 1 year of the target age after processing.")
                return {}

            # Create a DataFrame from the filtered data
            df_filtered = pd.DataFrame({
                'Title': book_titles,
                'similarity': similarities
            })

            # Sort by similarity in descending order (closest to 1 first)
            df_sorted = df_filtered.sort_values(by='similarity', ascending=False)

            # Select top 20 entries (or fewer if less than 20 match)
            top_20 = df_sorted.head(20)

            # Create a dictionary mapping book titles to similarity scores
            Output = ""
            result = dict(zip(top_20['Title'], top_20['similarity']))
            for i, (title, similarity) in enumerate(result.items(), 1):
                Output += f"{i}. {title} (Similarity: {similarity:.4f})"
            return result

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return {}
        except Exception as e:
            print(f"Error processing the file: {e}")
            return {}

    # # Example usage
    # if __name__ == "__main__":
    #     # Example input vector and age
    #     sample_emotion_vector = {'Anger': 0.014299376616268058, 'Anticipation': 0.063514265300745, 'Disgust': 0.00892495735446797, 'Fear': 0.03217397280838508, 'Joy': 0.07946445720684969, 'Sadness': 0.027790438634022925, 'Surprise': 0.028688857354523756, 'Trust': 0.10314579591498067, 'Objective': 0.6419978788097568}
    #     sample_age = 16

    #     # Call the function
    #     top_books = find_top_similar_books(sample_emotion_vector, sample_age)

    #     # Display results
    #     if top_books:
    #         print("\nTop matching books:")
    #         for i, (title, similarity) in enumerate(top_books.items(), 1):
    #             print(f"{i}. {title} (Similarity: {similarity:.4f})")
    #     else:
    #         print("No matching books found.")

with open("Conclusive_Data_Llama.csv") as inf:
    with open("Final_Conclusive_Llama.txt", "w") as of:
        ages = [15, 16, 13, 14, 15, 16, 17, 18, 13, 14]
        Llama_Object = Llama_Test()
        for i, line in enumerate(inf.readlines()):
            if line[0] == "{":
                index = ages[int(i/2)]
                print(index)
                vector = ast.literal_eval(line)
                print(vector)
                of.write(str(Llama_Object.find_top_similar_books(vector, ages[int(i/2)])) + "\n")
            else:
                of.write(line)

# print(Llama_Test.Cleanup_Description(r"One thousand years after a cataclysmic event leaves humanity on the brink of extinction, the\n survivors take refuge in continuums designed to sustain the human race until repopulation of Earth becomes possible. Against this backdrop, a group of yinuums designed to sustain the human race until repopulation of Earth becomes possible. Against this backdrop, a group of young friends in \nthe underwater Thirteenth Continuum dream about life outside their totalitarian existence, an idea that has been outlawed for centuries. When a shocking discovery turns the dream into a reality, they must decide if they will risk their own extinction to experience something no one has for generations, the Surface-- Provided by publisher."))
# Llama_Test.Filter_Sample(r"C:\Users\natha\Programming\LLM_Research\LLM_Semantics\Code\complete_pl_output.csv", 12, 19)
# Llama_Test.Recommend_for_Sample("Sample_Data.txt", "LLM_Recommendations.txt")
# Llama_Test.Convert_to_Vectors("LLM_Recommendations.txt", "Calculated_Emotion_Vectors.csv")
# Llama_Test.Determine_Data_Difference("Calculated_Emotion_Vectors.csv", "Results.txt")
# Test_Object = Llama_Test()
# print(Test_Object.get_recommended_books("Harry Potter and the Sorcerer's Stone"))
# ev1 = {'Anger': 0.04736088869499523, 'Anticipation': 0.0941099483403472, 'Disgust': 0.03619899872845963, 'Fear': 0.0443336649624124, 'Joy': 0.08280441155346056, 'Sadness': 0.02848463245034874, 'Surprise': 0.032543985188416745, 'Trust': 0.0968179569165944, 'Objective': 0.537345513164965}
# ev2 = {'Anger': 0.0, 'Anticipation': 0.09648702533112835, 'Disgust': 0.0, 'Fear': 0.021374588504397014, 'Joy': 0.10542787815093578, 'Sadness': 0.01169188445667125, 'Surprise': 0.03780350215600513, 'Trust': 0.10127814784476762, 'Objective': 0.6259369735560947}
# print(Llama_Test.find_cosine_similarity(ev1, ev2))

