import numpy as np
import warnings


class Encoder:
    def __init__(self) -> None:
        self.outcome_mapping = {0: "died", 1: "euthanized", 2: "lived"}
        self.encoded_len = 89
        self.encode_order = {
            "rectal_temp": ["numeric"],
            "pulse": ["numeric"],
            "respiratory_rate": ["numeric"],
            "nasogastric_reflux_ph": ["numeric"],
            "packed_cell_volume": ["numeric"],
            "total_protein": ["numeric"],
            "abdomo_protein": ["numeric"],
            "abdomen": ["distend_large", "distend_small", "firm", "normal", "other"],
            "abdominal_distention": ["moderate", "none", "severe", "slight"],
            "abdomo_appearance": ["clear", "cloudy", "serosanguious"],
            "age": ["adult", "young"],
            "capillary_refill_time": ["3", "less_3_sec", "more_3_sec"],
            "cp_data": ["no", "yes"],
            "mucous_membrane": [
                "bright_pink",
                "bright_red",
                "dark_cyanotic",
                "normal_pink",
                "pale_cyanotic",
                "pale_pink",
            ],
            "nasogastric_reflux": ["less_1_liter", "more_1_liter", "none"],
            "nasogastric_tube": ["none", "significant", "slight"],
            "pain": ["alert", "depressed", "extreme_pain", "mild_pain", "severe_pain"],
            "peripheral_pulse": ["absent", "increased", "normal", "reduced"],
            "peristalsis": ["absent", "hypermotile", "hypomotile", "normal"],
            "rectal_exam_feces": ["absent", "decreased", "increased", "normal"],
            "surgery": ["no", "yes"],
            "surgical_lesion": ["no", "yes"],
            "temp_of_extremities": ["cold", "cool", "normal", "warm"],
            "lesion_1": [
                "gastric",
                "sm_intestine",
                "lg_colon",
                "lg_colon_and_cecum",
                "cecum",
                "transverse_colon",
                "retum/descending_colon",
                "uterus",
                "bladder",
                "all_intestinal_sites",
                "simple",
                "strangulation",
                "inflammation",
                "other",
                "mechanical",
                "paralytic",
                "obturation",
                "intrinsic",
                "extrinsic",
                "adynamic",
                "volvulus/torsion",
                "intussuption",
                "thromboembolic",
                "hernia",
                "lipoma/slenic_incarceration",
                "displacement",
            ],
        }

    def get_features_dict(self):
        return self.encode_order

    def encode_records(self, records: list) -> np.array:
        encoded_records = None
        for record in records:
            encoded = self.encode_one_record(record)

            if encoded_records is None:
                encoded_records = encoded
                continue

            encoded_records = np.vstack((encoded_records, encoded))
        return encoded_records

    def encode_one_record(self, record: dict) -> np.array:
        encoded_record = np.array([])
        for key, possible_values in self.encode_order.items():
            actual_value = record.get(key)

            if actual_value is None:
                encoded_value = self.__encode_null_elem(key, possible_values)
            elif possible_values[0] == "numeric":
                encoded_value = self.__encode_numeric_elem(
                    key, possible_values, actual_value
                )
            elif key == "lesion_1":
                encoded_value = self.__encode_lesion(
                    key, possible_values, str(actual_value)
                )
            else:
                encoded_value = self.__encode_categorical_elem(
                    key, possible_values, actual_value
                )

            assert len(encoded_value) != 0, "Something went wrong, {key} wast encoded."

            encoded_record = np.append(encoded_record, encoded_value)

        assert (
            len(encoded_record) == self.encoded_len
        ), "Something went wrong, try another input"
        return encoded_record

    def __encode_null_elem(self, key: str, possible_values: list) -> np.array:
        default_values = {
            "rectal_temp": [37.8],
            "pulse": [73],
            "respiratory_rate": [9],
            "nasogastric_reflux_ph": [4],
            "packed_cell_volume": [None],
            "total_protein": [7],
            "abdomo_protein": [2],
            "abdomen": [0, 0, 0, 1, 0],
            "abdominal_distention": [0, 1, 0, 0],
            "abdomo_appearance": [None] * 3,
            "age": [1, 0],
            "capillary_refill_time": [None] * 3,
            "cp_data": [1, 0],
            "mucous_membrane": [0, 0, 0, 1, 0, 0],
            "nasogastric_reflux": [1, 0, 0],
            "nasogastric_tube": [0, 0, 1],
            "pain": [1, 0, 0, 0, 0],
            "peripheral_pulse": [0, 0, 1, 0],
            "peristalsis": [0, 0, 0, 1],
            "rectal_exam_feces": [None] * 4,
            "surgery": [1, 0],
            "surgical_lesion": [1, 0],
            "temp_of_extremities": [None] * 4,
            "lesion_1": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }
        warnings.warn(f"{key}: Value was missed.")
        encoded = np.array(default_values[key])
        return encoded

    def __encode_numeric_elem(
        self, key: str, possible_values: list, actual_value
    ) -> np.array:
        try:
            encoded = np.array([float(actual_value)])
        except ValueError:
            warnings.warn(f"Wrong format, {key} must be numeric! Value was skiped.")
            encoded = np.zeros(len(possible_values)).astype(np.float16)
        return encoded

    def __encode_categorical_elem(
        self, key: str, posible_values: list, actual_value: str
    ) -> np.array:
        """
        possible_values: [normal, not_normal, smth]
        actual_value: not_normal
        return: [0, 1, 0]
        """
        posible_values = np.array(posible_values)
        encoded = np.in1d(posible_values, actual_value).astype(np.float16)
        if not any(encoded):
            warnings.warn(f" {key}: Wrong format, value was skiped.")
        return encoded

    def __encode_lesion(
        self, key: str, possible_values: list, actual_value: str
    ) -> np.array:
        posible_values = np.array(possible_values)
        relevant_values = np.array(self.__split_lesion(actual_value))
        encoded = np.in1d(posible_values, relevant_values).astype(np.float16)
        if not any(encoded):
            warnings.warn(f" {key}: Wrong format, value was skiped.")
        return encoded

    def __split_lesion(self, code: str) -> list:
        """
        convert code 2209 into [sm_intestine, strangulation, none, lipoma/slenic_incarceration]
        """
        lesion_site = {
            "1": "gastric",
            "2": "sm_intestine",
            "3": "lg_colon",
            "4": "lg_colon_and_cecum",
            "5": "cecum",
            "6": "transverse_colon",
            "7": "retum/descending_colon",
            "8": "uterus",
            "9": "bladder",
            "11": "all_intestinal_sites",
            "00": "none",
        }
        lesion_type = {
            "1": "simple",
            "2": "strangulation",
            "3": "inflammation",
            "4": "other",
            "0": "none",
        }
        lesion_subtype = {
            "1": "mechanical",
            "2": "paralytic",
            "0": "none",
        }
        lesion_specific_code = {
            "1": "obturation",
            "2": "intrinsic",
            "3": "extrinsic",
            "4": "adynamic",
            "5": "volvulus/torsion",
            "6": "intussuption",
            "7": "thromboembolic",
            "8": "hernia",
            "9": "lipoma/slenic_incarceration",
            "10": "displacement",
            "0": "none",
        }
        if code[:2] in ["00", "11"]:
            site, type, subtype, s_code = code[:2], code[2], code[3], code[4:]
        elif code[0] == "0":
            site, type, subtype, s_code = "none", "none", "none", "none"
        else:
            site, type, subtype, s_code = code[0], code[1], code[2], code[3:]

        return [
            lesion_site.get(site, "none"),
            lesion_type.get(type, "none"),
            lesion_subtype.get(subtype, "none"),
            lesion_specific_code.get(s_code, "none"),
        ]
