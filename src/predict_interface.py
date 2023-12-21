import pickle
from pathlib import Path

from models.model import Model


# TODO: убрать за ненадобностью
if __name__ == "__main__":
    # формат инпута
    records = [
        {
            "surgery": "yes",
            "age": "adult",
            "rectal_temp": 38.1,
            "pulse": 132.0,
            "respiratory_rate": 24.0,
            "temp_of_extremities": "cool",
            "peripheral_pulse": "reduced",
            "mucous_membrane": "dark_cyanotic",
            "capillary_refill_time": "more_3_sec",
            "pain": "depressed",
            "peristalsis": "absent",
            "abdominal_distention": "slight",
            "nasogastric_tube": "slight",
            "nasogastric_reflux": "less_1_liter",
            "nasogastric_reflux_ph": 6.5,
            "rectal_exam_feces": "decreased",
            "abdomen": "distend_small",
            "packed_cell_volume": 57.0,
            "total_protein": 8.5,
            "abdomo_appearance": "serosanguious",
            "abdomo_protein": 3.4,
            "surgical_lesion": "yes",
            "lesion_1": 2209,
            "cp_data": "no",
        },
        {
            "surgery": "yes",
            "age": "adult",
            "rectal_temp": 37.5,
            "pulse": 88.0,
            "respiratory_rate": 12.0,
            "temp_of_extremities": "cool",
            "peripheral_pulse": "normal",
            "mucous_membrane": "pale_cyanotic",
            "capillary_refill_time": "more_3_sec",
            "pain": "mild_pain",
            "peristalsis": "absent",
            "abdominal_distention": "moderate",
            "nasogastric_tube": "none",
            "nasogastric_reflux": "more_1_liter",
            "nasogastric_reflux_ph": 2.0,
            "rectal_exam_feces": "absent",
            "abdomen": "distend_small",
            "packed_cell_volume": 33.0,
            "total_protein": 64.0,
            "abdomo_appearance": "serosanguious",
            "abdomo_protein": 2.0,
            "surgical_lesion": "yes",
            "lesion_1": 2208,
            "cp_data": "no",
        },
    ]

    # загружаем модель (model_path возможно надо будет подправить)
    model_path = Path().absolute().parent / "models" / "xgb.pickle"
    xgb = pickle.load(open(model_path, "rb"))

    model = Model(xgb)

    # Можно передавать список из словарей
    classes = model.predict(records)
    print(classes)  # ['died', 'euthanized']

    # Можно передавать одночный словарь
    one_class = model.predict(records[0])
    print(one_class)  # ['died']

    # Можно получить метки классов
    classes = model.predict(records, class_names=False)
    print(classes)  # [0, 1]
