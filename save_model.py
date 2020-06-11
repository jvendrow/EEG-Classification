import os
import pickle, json
import os.path
from os.path import dirname, join

def replace_model_if_better(old_model_path, new_acc, new_model, config, history=None):
    if os.path.isfile(old_model_path):
        print("Old model exists. Comparing performance.")
        f = open(old_model_path, 'rb')
        model_dict = pickle.load(f)
        f.close()
        if new_acc > model_dict['acc']:
            print("New model is better than the old one. Replacing the old model with the new model.")
            os.remove(old_model_path)
            f = open(old_model_path, 'wb')
            model_dict = {'acc': new_acc, 'model': new_model}
            if history is not None:
                model_dict['history'] = history
            pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            folder_name = dirname(old_model_path)
            config_path = join(folder_name, 'config.json')
            with open(config_path, 'w') as fp:
                json.dump(config, fp, indent=4)
            return True
        else:
            print("New model is worse than the old one. Will not update the old model")
            return False
    else:
        print("No existing model in specified path. Saving the new model")
        f = open(old_model_path, 'wb')
        model_dict = dict()
        model_dict['acc'] = new_acc
        model_dict['model'] = new_model
        if history is not None:
            model_dict['history'] = history
        pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        folder_name = dirname(old_model_path)
        config_path = join(folder_name, 'config.json')
        with open(config_path, 'w') as fp:
            json.dump(config, fp, indent=4)
        return True
