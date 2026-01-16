from models.SGC import CombinedLearner
def get_model(model_name, args):
    name = model_name.lower()
    if name == "playground":
        from models.clgcbm import Player
        return Player(args)
    elif name == "combined_learner": 
        return CombinedLearner(args)
    else:
        assert 0
