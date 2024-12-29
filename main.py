from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np    
import uvicorn
import os

# Initialize the FastAPI app
app = FastAPI()

class MushroomClassifierHandler:

    def __init__(self, model_path, species_path):
        self.toxicity_classifier = joblib.load(model_path)
        self.species_classifier = joblib.load(species_path)
        self.mushrooms_species_df_full = pd.read_csv('MushroomDataset/primary_data.csv', sep=';')
        self.toxicity_data = None
        self.species_data = None
        self.habitat_dict = {'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths', 'h': 'heaths', 'u': 'urban', 'w': 'waste', 'd': 'woods'}
        self.season_dict = {'s': 'spring', 'u': 'summer', 'a': 'autumn', 'w': 'winter'}

        self.load_toxicity_data()
        self.load_species_data()

    def __call__(self, input):
        """
        Predicts the toxicity and species of the mushroom based on the input data
        Returns a dictionary with the predictions
        """
        # convert input to dataframe
        input = pd.DataFrame([input])
        # encode the input
        input_encoded = self.encode_input(input, self.toxicity_data)
        species_encoded = self.encode_input(input, self.toxicity_data)
 
        # get predictions
        toxicity_prediction = self.get_toxicity_prediction(input_encoded)
        species_prediction = self.get_species_prediction(species_encoded)

        return {
            'toxicity': toxicity_prediction,
            'species': species_prediction
        }
    

    def load_toxicity_data(self):
        """
        Loads the toxicity data from the dataset
        """
        mushrooms_df = pd.read_csv('MushroomDataset/secondary_data.csv', sep=';')
        mushrooms_df.drop(["class", "veil-color", 'veil-type', 'gill-attachment', 'has-ring', 'does-bruise-or-bleed', 'season', 'habitat', 'spore-print-color', 'stem-width', 'cap-diameter', 'stem-height'], axis=1, inplace=True) # remove unnecessary and unvisualizable features
        self.toxicity_data = pd.get_dummies(mushrooms_df, drop_first=True)
    

    def load_species_data(self):
        """
        Loads the species data from the dataset
        """
        mushrooms_df = pd.read_csv('MushroomDataset/secondary_data.csv', sep=';')
        mushrooms_df.drop(["class", "veil-color", 'veil-type', 'gill-attachment', 'has-ring', 'does-bruise-or-bleed', 'season', 'habitat', 'spore-print-color', 'stem-width', 'cap-diameter', 'stem-height'], axis=1, inplace=True) # remove unnecessary and unvisualizable features
        self.species_data = pd.get_dummies(mushrooms_df, drop_first=True)


    def encode_input(self, input, reference_data):
        """ 
        Encodes the input data into the same format as the reference data
        """
        
        # encode the input
        input_encoded = pd.get_dummies(input, drop_first=False)
        # add missing columns with False
        for col in reference_data.columns:
            if col not in input_encoded.columns:
                input_encoded[col] = False

        # reorder columns
        input_encoded = input_encoded[reference_data.columns]

        return input_encoded


    def get_toxicity_prediction(self, input):
        """ 
        Predicts the toxicity of the mushroom based on the input data
        Returns 0 for edible and 1 for poisonous
        """

        #toxicity_prediction = self.toxicity_classifier(torch.tensor(input.values, dtype=torch.float32)).detach().numpy()
        toxicity_prediction = self.toxicity_classifier.predict(input)
        # round the result
        toxicity_prediction = np.round(toxicity_prediction).flatten()[0]
        return int(toxicity_prediction) # edible = 0, poisonous = 1


    def get_species_prediction(self, input):
        """
        Predicts the species of the mushroom based on the input data
        Returns a dictionary with more information about the species
        """

        species_prediction = self.species_classifier.predict(input)
        # get more information about the species 
        species_info_df = self.mushrooms_species_df_full.loc[self.mushrooms_species_df_full.loc[:, 'name'] == species_prediction[0]]
        # get the habitat and season information in the proper format
        habitat = species_info_df['habitat'].values[0].replace(' ', '').strip('[]').split(',')
        habitat = [self.habitat_dict[hab] for hab in habitat]
        season = species_info_df['season'].values[0].replace(' ', '').strip('[]').split(',')
        season = [self.season_dict[s] for s in season]
        # get toxicity
        toxicity = species_info_df['class'].values[0]

        species_info = {
            'family': species_info_df['family'].values[0],
            'name': species_info_df['name'].values[0],
            'class': toxicity,
            'season': season,
            'habitat': habitat
        }

        return species_info

# Initialize the handler
handler = MushroomClassifierHandler('models/toxicity_tree_classifier.joblib', 'models/species_classifier.joblib')

# Define endpoints for predictions
@app.post("/predict")
def predict(input: dict):
    try:
        # Prepare data for the model
        return handler(input)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    # Get the PORT from the environment variable (default to 8000 if not set)
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# running the app in CLI:
# uvicorn app:app --reload --host 0.0.0.0 --port 8000