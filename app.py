from flask import Flask, render_template, request, send_from_directory, url_for, send_file
from Predict import Predict
import pandas as pd
import datetime
from os import path

def count_files(request):
  imagefiles = request.files.getlist('imagefile')
  num_files = len(imagefiles)
  files_path = []
  for imagefile in imagefiles:
    file_name = './files/' + imagefile.filename
    imagefile.save(file_name)
    files_path.append(file_name)
  return num_files, files_path

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/download/<string:name_file>', methods=['GET', 'POST'])
def download_file(name_file=''):
    print(name_file)
    url_file = path.join('./predictions_files/', name_file)
    resp = send_file(url_file, as_attachment=True)
    return resp

@app.route('/', methods=['POST'])
def predictions():
    _, files_path = count_files(request)
    obj_predict = Predict()
    
    results = ''
    files = []
    animal = []
    accuracy_rate = []
    
    for file_path in files_path:
        if ".png" in file_path or ".jpg" in file_path or ".jpeg" in file_path:  
            labels = obj_predict.predict_img(file_path)
        
            name = ''
            percentage = 0
            for i in labels:
                if i[1] > percentage:
                    name = i[0][3:]
                    percentage = i[1]
            
            files.append(file_path.replace('./files/',''))
            animal.append(name)
            accuracy_rate.append(round(float(percentage), 2))
              
        elif ".mp4" in file_path:
            name, percentage = obj_predict.predict_video(file_path)
            
            files.append(file_path.replace('./files/',''))
            animal.append(name)
            accuracy_rate.append(round(float(percentage), 2))
            
    print(results)
    
    df = pd.DataFrame({'Archivo': files,
                       'Animal': animal,
                       'Porcentaje de precisión': accuracy_rate})
    df = df[['Archivo', 'Animal', 'Porcentaje de precisión']]
    
    now = datetime.datetime.now()
    file_xlsx_name = now.strftime("%Y-%m-%d_%H-%M-%S.xlsx")
    
    writer = pd.ExcelWriter('./predictions_files/' + file_xlsx_name)
    df.to_excel(writer, 'Hoja de datos', index=False)
    writer.save()
    
    # print(file_xlsx_name)
    
    return render_template('index.html', download_link=url_for('download_file', name_file=file_xlsx_name))
    # return render_template('index.html', prediction=num_files)
    
if __name__ == '__main__':
    path_file_remove = './files/'
    path_predictions_remove = './predictions_files/'
    #app.run(port=3000, debug=True)
    app.run(port=3000, debug=True, use_reloader=False)