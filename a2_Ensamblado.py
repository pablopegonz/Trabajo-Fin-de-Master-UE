from ultralytics import YOLO

from PIL import Image
from pathlib import Path

import os
import time


class Arbitrator ():

    def __init__(self) -> None:
        
        self.model = YOLO('./modelos/00_pred_ue.pt')
        self.cls = YOLO('./modelos/00_cls.pt')

        self.project_n = 'shared_folder'
        self.name_n = 'temp'

        self.img_og_path = None

        self.img_cropped_paths = None
        self.decisions = None


    def arbitro(self):

        i = 0
        # ARBITRO
        for decision in self.decisions:

            i += 1

            if int(decision["clase-detc"]) == 0 and int(decision["clase-cls"]) == 0:
                clase_final = 1
                conf_to_drone = ((decision["conf-detc"] * 0.35) + (decision["conf-cls"] * 0.65))
            else:
                clase_final = 0
                conf_to_drone = (1 - ((decision["conf-detc"] * 0.35) + (decision["conf-cls"] * 0.65)))

            self.final_decision = clase_final
            self.decisions = {}

            return clase_final, conf_to_drone

           


    def clasificador_dron(self):

        # CLASIFICACION
        try:

            for decision in self.decisions:

                img = Image.open(decision["path"])
                results = self.cls.predict(source=img, save=True)

                for r in results:

                    save_dir = r.save_dir

                    confidence_list = r.probs.data.tolist()

                    confidence = max(confidence_list)
                    family = confidence_list.index(confidence)

                    decision["clase-cls"] = family
                    decision["conf-cls"] = confidence
                    decision["path-cls"] = save_dir
            return self.arbitro()           

        except Exception as e:
            print(f"Error al procesar la imagen {decision["path"]}: {e}")
            
            


    def prediction_sis (self, raw_image_path):

        # PREDICCION
        try:
            self.img_og_path = raw_image_path

            img_og = Image.open(self.img_og_path)
            time.sleep(0.1)

            results = self.model.predict(source=img_og, save_crop=True, project=self.project_n, name=self.name_n)

            img_cropped_paths = []
            pre_cropped_paths = []
            decisions = []

            for r in results:
                classes_detected = r.boxes.cls.tolist()
                confidences = r.boxes.conf.tolist()

                for index, num_class_detected in enumerate(classes_detected):
                    class_detected_ind = int(classes_detected[index])
                    class_detected_name = r.names[num_class_detected]
                    confidence = confidences[index]

                    decision = {"clase-detc": class_detected_ind, "conf-detc": confidence}
                    print(decision)
                    decisions.append(decision)

                    crop_path_i = os.path.join(r.save_dir, 'crops', class_detected_name)
                    pre_cropped_paths.append(crop_path_i)
                    pre_cropped_paths = list(set(pre_cropped_paths))

            
            for directorio in pre_cropped_paths:

                archivos = os.listdir(directorio)
                for archivo in archivos:
                    final_path = os.path.join(directorio, archivo)
                    img_cropped_paths.append(final_path)

            for diccionario, path in zip(decisions, img_cropped_paths):
                diccionario["path"] = path

            self.img_cropped_paths = img_cropped_paths
            self.decisions = decisions

            if not self.decisions:
                c = 9
                p = 9
                return c, p
            else:

                return self.clasificador_dron()

        except Exception as e:
            print(f"Error al procesar la imagen {self.img_og_path}: {e}")
        

   


if __name__ == "__main__":

    path = "./imagenes/0000005.jpg"
    test_dir_path = "./imagenes"
    predicciones = []

    arbitro = Arbitrator()
    c, p = arbitro.prediction_sis(path)
    
