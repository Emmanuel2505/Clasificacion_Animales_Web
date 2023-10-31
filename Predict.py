from PIL import Image
from torchvision import models, transforms
import torch
import timm
import cv2

class Predict:
    def __init__(self):
        # Crear un modelo ConvNext
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Utilizando el dispositivo {self.device}.")
        self.num_classes = 4
        self.model = timm.create_model('convnext_small_in22k', pretrained=True, num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load('./model-17.pt', map_location=torch.device('cpu')))
        
    def predict_img(self, image_path):
        # Transformar la imagen de entrada mediante redimensionamiento y normalización
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        # Cargar la imagen, preprocesarla y realizar predicciones
        img = Image.open(image_path)
        batch_t = torch.unsqueeze(transform(img), 0)
        self.model.eval()
        out = self.model(batch_t)

        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]

        # Devolver las 5 mejores predicciones ordenadas por las probabilidades más altas
        prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
        _, indices = torch.sort(out, descending=True)
        return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
    
    def predict_video(self, image_path):
        cap = cv2.VideoCapture(image_path)
        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        
        while cap.isOpened():
            status, frame = cap.read()
            
            if not status:
                break
            labels = []
            percentages = []
            
            frame = cv2.resize(frame, (256, 256)) # Cambiar de tamaño a 256
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Cambiar el canal a RGB
            frame = frame/255.0 # Normalizar los valores entre 0 y 1
            frame = torch.from_numpy(frame).float() # Convertir a tensor de PyTorch
            frame = frame.permute(2, 0, 1) # Cambiar el orden de las dimensiones a CxHxW
            frame = frame.unsqueeze(0) # Añadir una dimension extra para el lote
            output = self.model(frame)
            
            label =  torch.argmax(output).item()
            # #print(classes[label])
            preds = self.model(frame)
            print(preds)
            
            # #probability = torch.max(output).item()
            probability = torch.nn.functional.softmax(output, dim=1)[0] * 100
            percentage = probability[label].item()
            percentages.append(percentage if (percentage > 90) else 0.0)
            labels.append(classes[label] if (percentage > 90) else "No se esta seguro")
            
            break
            
            #preds = model(frame)
            #print(preds)
            # cv2.imshow("frame", frame)

            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #   break
        cap.release()
        characters_labels = "'[]0124,"
        characters_percentages = "[]"
        labels = "".join(x for x in labels if x not in characters_labels)
        # labels = labels.replace(' ','')
        labels = labels.replace(',','')
        labels = labels.replace('0','')
        labels = labels.replace('1','')
        labels = labels.replace('2','')
        labels = labels.replace('3','')
        percentages = str(percentages)
        percentages = "".join(x for x in percentages if x not in characters_percentages)
        #print(labels)
        #print(percentages)
        
        return labels, percentages