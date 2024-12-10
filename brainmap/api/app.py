from fastapi import FASTAPI

app = FASTAPI()


model = app.state.model


@app.get('/')
async def root():
    return {'message': 'Welcome to brainmap'}


@app.get('/brainmap/api/tumor')
async def tumor_result(image: global):
    return {'The patient result shows':f'{y.predict(image)* 100} % malignant'}
