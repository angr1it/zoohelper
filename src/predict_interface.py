import numpy as np
import warnings

class Encoder:
    def __init__(self) -> None:
        self.outcome_mapping = {0: 'died', 1: 'euthanized', 2: 'lived'} 
        self.encoded_len = 89        
        self.encode_order = {
                            'rectal_temp': ['numeric'],
                            'pulse': ['numeric'],
                            'respiratory_rate': ['numeric'],
                            'nasogastric_reflux_ph': ['numeric'],
                            'packed_cell_volume': ['numeric'],
                            'total_protein': ['numeric'],
                            'abdomo_protein': ['numeric'],
                            'abdomen': ['distend_large', 'distend_small', 'firm', 'normal', 'other'],
                            'abdominal_distention': ['moderate', 'none', 'severe', 'slight'],
                            'abdomo_appearance': ['clear', 'cloudy', 'serosanguious'],
                            'age': ['adult', 'young'],
                            'capillary_refill_time': ['3', 'less_3_sec', 'more_3_sec'],      
                            'cp_data': ['no', 'yes'],
                            'mucous_membrane': ['bright_pink', 'bright_red', 'dark_cyanotic', 'normal_pink', 
                                                'pale_cyanotic', 'pale_pink'],   
                            'nasogastric_reflux': ['less_1_liter', 'more_1_liter', 'none'],
                            'nasogastric_tube': ['none', 'significant', 'slight'],
                            'pain': ['alert', 'depressed', 'extreme_pain', 'mild_pain', 'severe_pain'],
                            'peripheral_pulse': ['absent', 'increased', 'normal', 'reduced'],
                            'peristalsis': ['absent', 'hypermotile', 'hypomotile', 'normal'],
                            'rectal_exam_feces': ['absent', 'decreased', 'increased', 'normal'],
                            'surgery': ['no', 'yes'],
                            'surgical_lesion': ['no', 'yes'],
                            'temp_of_extremities': ['cold', 'cool', 'normal', 'warm'],
                            'lesion_1': ["gastric",  "sm_intestine", "lg_colon", "lg_colon_and_cecum", "cecum", 
                                         "transverse_colon","retum/descending_colon", "uterus", "bladder",
                                         "all_intestinal_sites", "simple", "strangulation","inflammation", "other", 
                                         "mechanical", "paralytic", "obturation", "intrinsic", "extrinsic", "adynamic", 
                                         "volvulus/torsion", "intussuption", "thromboembolic", "hernia", 
                                         "lipoma/slenic_incarceration", "displacement"] 
                            }

    def encode_records(self, records: list) -> np.array:
        encoded_records = None
        for record in records:
            encoded = self.encode_one_record(record)

            if encoded_records is None:
                encoded_records = encoded
                continue
            
            encoded_records=np.vstack((encoded_records, encoded))
        return encoded_records
                  

    def encode_one_record(self, record: dict) -> np.array:
        encoded_record = np.array([])
        for key, possible_values in self.encode_order.items():
            actual_value = record.get(key)
            
            if actual_value is None:
                encoded_value = self.__encode_null_elem(key, possible_values)
            elif possible_values[0] == 'numeric':
                encoded_value = self.__encode_numeric_elem(key, possible_values, actual_value)
            elif key == 'lesion_1':
                encoded_value = self.__encode_lesion(key, possible_values, str(actual_value))
            else:
                encoded_value = self.__encode_categorical_elem(key, possible_values, actual_value)

            assert len(encoded_value) != 0, 'Something went wrong, {key} wast encoded.'

            encoded_record = np.append(encoded_record, encoded_value)
        
        assert len(encoded_record) == self.encoded_len, 'Something went wrong, try another input'
        return encoded_record
    
            
    def __encode_null_elem(self, key: str, possible_values: list) -> np.array:
        warnings.warn(f'{key}: Value was missed.')
        encoded = np.zeros(len(possible_values))
        return encoded


    def __encode_numeric_elem(self, key: str, possible_values: list, actual_value) -> np.array:
        try:
            encoded = np.array([float(actual_value)])
        except ValueError:
            warnings.warn(f'Wrong format, {key} must be numeric! Value was skiped.')
            encoded = np.zeros(len(possible_values)).astype(np.float16)
        return encoded


    def __encode_categorical_elem(self, key: str, posible_values: list, actual_value: str) -> np.array:
        """
        possible_values: [normal, not_normal, smth]
        actual_value: not_normal
        return: [0, 1, 0]
        """
        posible_values = np.array(posible_values)
        encoded = np.in1d(posible_values, actual_value).astype(np.float16)
        if not any(encoded):
            warnings.warn(f' {key}: Wrong format, value was skiped.')
        return encoded
    

    def __encode_lesion(self, key: str, possible_values: list, actual_value: str) -> np.array:
        posible_values = np.array(possible_values) 
        relevant_values = np.array(self.__split_lesion(actual_value))
        encoded = np.in1d(posible_values, relevant_values).astype(np.float16)
        if not any(encoded):
            warnings.warn(f' {key}: Wrong format, value was skiped.')
        return encoded
        

    def __split_lesion(self, code: str) -> list:
        """
        convert code 2209 into [sm_intestine, strangulation, none, lipoma/slenic_incarceration]
        """
        lesion_site = {
                "1": "gastric",  "2": "sm_intestine", "3": "lg_colon",
                "4": "lg_colon_and_cecum", "5": "cecum", "6": "transverse_colon",
                "7": "retum/descending_colon", "8": "uterus", "9": "bladder",
                "11": "all_intestinal_sites", "00": "none",
        }
        lesion_type = {
                "1": "simple", "2": "strangulation",
                "3": "inflammation", "4": "other",
                "0": "none",
                }
        lesion_subtype = {
                "1": "mechanical", "2": "paralytic", "0": "none",
                }
        lesion_specific_code = {
                "1": "obturation", "2": "intrinsic", "3": "extrinsic",
                "4": "adynamic", "5": "volvulus/torsion", "6": "intussuption",
                "7": "thromboembolic", "8": "hernia", "9": "lipoma/slenic_incarceration",
                "10": "displacement", "0": "none",
                }
        if code[:2] in ['00', '11']:
            site, type, subtype, s_code = code[:2], code[2], code[3], code[4:]
        elif code[0] == '0':
            site, type, subtype, s_code = 'none', 'none', 'none', 'none'
        else:
            site, type, subtype, s_code = code[0], code[1], code[2], code[3:]

        return [lesion_site.get(site, "none"), lesion_type.get(type, "none"), 
                lesion_subtype.get(subtype, "none"), lesion_specific_code.get(s_code, "none")]
    


class Model:
    def __init__(self, model=None) -> None:
        self.encoder = Encoder()
        self.model = model
    
    def predict(self, X, class_names=True):
        if self.model == None:
            warnings.warn('Model is not defined! Load the model first')
            return
        
        if type(X) == dict:
            encoded_X = self.encoder.encode_one_record(X).reshape((1, -1))
        elif type(X) == list:
            encoded_X = self.encoder.encode_records(X)
        else:
            raise ValueError("X must be list or dict")
        
        try:
            y = self.model.predict(encoded_X)
        except AttributeError:
            raise AttributeError('Model must have predict method')
        
        if class_names:
            y=[self.encoder.outcome_mapping[i] for i in y]

        return y
    


if __name__ == '__main__':
    import pickle
    from pathlib import Path
    
    # формат инпута
    records = [
    {'surgery': 'yes', 'age': 'adult', 'rectal_temp': 38.1, 'pulse': 132.0,
     'respiratory_rate': 24.0, 'temp_of_extremities': 'cool',
     'peripheral_pulse': 'reduced', 'mucous_membrane': 'dark_cyanotic',
     'capillary_refill_time': 'more_3_sec', 'pain': 'depressed', 
     'peristalsis': 'absent', 'abdominal_distention': 'slight', 
     'nasogastric_tube': 'slight','nasogastric_reflux': 'less_1_liter', 
     'nasogastric_reflux_ph': 6.5, 'rectal_exam_feces': 'decreased', 
     'abdomen': 'distend_small', 'packed_cell_volume': 57.0, 'total_protein': 8.5, 
     'abdomo_appearance': 'serosanguious', 'abdomo_protein': 3.4, 
     'surgical_lesion': 'yes', 'lesion_1': 2209, 'cp_data': 'no'}, 
     
    {'surgery': 'yes', 'age': 'adult', 'rectal_temp': 37.5, 'pulse': 88.0,
     'respiratory_rate': 12.0, 'temp_of_extremities': 'cool', 
     'peripheral_pulse': 'normal', 'mucous_membrane': 'pale_cyanotic',
     'capillary_refill_time': 'more_3_sec', 'pain': 'mild_pain',
     'peristalsis': 'absent', 'abdominal_distention': 'moderate',
     'nasogastric_tube': 'none', 'nasogastric_reflux': 'more_1_liter',
     'nasogastric_reflux_ph': 2.0, 'rectal_exam_feces': 'absent',
     'abdomen': 'distend_small', 'packed_cell_volume': 33.0, 'total_protein': 64.0,
     'abdomo_appearance': 'serosanguious', 'abdomo_protein': 2.0, 
     'surgical_lesion': 'yes', 'lesion_1': 2208,'cp_data': 'no'}
    ]

    # загружаем модель (model_path возможно надо будет подправить)
    model_path = Path().absolute().parent/'models'/'xgb.pickle'
    xgb = pickle.load(open(model_path, 'rb'))

    model = Model(xgb)

    # Можно передавать список из словарей 
    classes = model.predict(records)
    print(classes)      # ['died', 'euthanized']

    # Можно передавать одночный словарь
    one_class = model.predict(records[0])
    print(one_class)    # ['died']

    # Можно получить метки классов
    classes = model.predict(records, class_names=False)
    print(classes)      # [0, 1]
