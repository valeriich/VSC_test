from main import BikeRentalsPredictionService
import pickle

brp = BikeRentalsPredictionService()

for i in ["casual", "registered"]:
    with open(f'model_{i}.pkl', 'rb') as f:
        model = pickle.load(f)

    brp.pack(f"model_{i}",  model)


brp.save()