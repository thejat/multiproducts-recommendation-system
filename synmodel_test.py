from synthetic_models.rcm_synthetic_model import generate_derived_rcm_choice_model
import pickle

with open('synthetic_models/models/tafeng.pkl','rb') as f:
    rcm_model = pickle.load(f)


generate_derived_rcm_choice_model(rcm_model, num_products=200)