from mtcnn import MTCNN
import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import numpy as np

import face_detector as fd
# import dlib
import cv2
import os
# from PIL import Image

export_file_url = 'https://drive.google.com/uc?export=download&id=1rpTBN-ImVr_Rar78no5R_k6df0tLRyTS'
export_file_name = 'emotion_gender.pkl'

basedir = os.path.abspath(os.path.dirname(__file__))
# predictor_path = os.path.join(basedir, 'shape_predictor_68_face_landmarks.dat')
# sp = dlib.shape_predictor(predictor_path)
detector = detector = MTCNN()

# UPLOAD_FOLDER = os.path.join(basedir, 'upload_images')



classes = ['female', 'happy', 'male', 'unhappy']
k_out = ['Female', 'Happy', 'Male', 'Not-happy']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = cv2.imdecode(np.fromstring(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    # prediction = learn.predict(img)[0]
    try:
        img= fd.detect_n_crop(img,detector)
        # cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'im.jpg'), img)
        #convert cv image to format expected by fastai
        t = torch.tensor(np.ascontiguousarray(img).transpose(2,0,1)).float()/255
        # make prediction
        _,_,outputs = learn.predict(Image(t))
        np_out = outputs.numpy()
        ind = np.argpartition(np_out, -2)[-2:]
        prediction=k_out[ind[0]]+' '+k_out[ind[1]]
    except:
        prediction='No face'
    
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
