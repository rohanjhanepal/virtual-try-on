import uvicorn
from fastapi import FastAPI, File, UploadFile , Request
from fastapi.responses import HTMLResponse , FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import aiofiles
import run
import base64
import os
app = FastAPI()


templates = Jinja2Templates(directory="templates")
app.mount("/static" , StaticFiles(directory="static") , name="static")
 

@app.get('/' , response_class=HTMLResponse)
def get_basic_form(request: Request):
    clothes = os.listdir('static/clothes')
    context = {
        'clothes_list': clothes,
    }
    return templates.TemplateResponse("index.html" , {"request":request})

@app.post('/' , response_class=HTMLResponse)
async def get_basic_form(request: Request , file:bytes = File(...),file1:bytes = File(...) ):
    #content = await file.read()

    async with aiofiles.open("lady.jpg", 'wb') as out_file:
        #content = await file.read()  # async read
        await out_file.write(file)
    async with aiofiles.open("cloth.jpg", 'wb') as out_file:
        #content = await file.read()  # async read
        await out_file.write(file1)
        
    run.execute()
   
    return templates.TemplateResponse("display.html" , {"request":request})
    

@app.get('/img_final/' , response_class=HTMLResponse)
async def get_image_final(request: Request):

    return './static/lady.jpg'



if __name__ == "__main__":
    uvicorn.run(app)