# Удаляем устаревший импорт, используем pd.read_csv
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from fastapi.staticfiles import StaticFiles
app = FastAPI()

app.mount("/index", StaticFiles(directory="static", html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/linear_regression")
async def linear_regression():
    try:
        # Проверяем, существует ли файл
        import os
        if not os.path.exists('data.csv'):
            return JSONResponse(
                status_code=404,
                content={"error": "Файл data.csv не найден. Сначала загрузите CSV файл."}
            )
        
        data = pd.read_csv('data.csv') # получаем наши данные
        
        # Проверяем наличие колонки target
        if 'target' not in data.columns:
            return JSONResponse(
                status_code=400,
                content={"error": "В CSV файле должна быть колонка 'target'"}
            )
        
        y = data['target']
        x = data.drop(columns=['target'])
        
        # Проверяем, что есть данные для анализа
        if len(x.columns) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Нет признаков для анализа (кроме колонки 'target')"}
            )
        
        x = StandardScaler().fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
        
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        # Конвертируем numpy arrays в списки для JSON
        y_pred_list = y_pred.tolist() if hasattr(y_pred,  'tolist') else list(y_pred)
        y_test_list = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
        
        mse = mean_squared_error(y_pred, y_test)
        
        result = pd.DataFrame({
            'prediction': y_pred_list,
            'correct': y_test_list,
            'MSE': [mse] * len(y_pred_list)
        })
        
        # Сохраняем результаты
        result.to_excel('result.xlsx', index=False)
        
        # Создаем график
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Реальные значения') 
        plt.ylabel('Предсказанные значения')
        plt.title('Линейная регрессия: Реальные vs Предсказанные значения')
        plt.savefig('result.png', dpi=150, bbox_inches='tight')
        plt.close()  # Закрываем фигуру для освобождения памяти
        
        return {
            "success": True,
            "MSE": mse,
            "message": "Анализ завершен. Файлы готовы для скачивания."
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка при выполнении линейной регрессии: {str(e)}"}
        )

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Проверяем тип файла
        if not file.filename.endswith('.csv'):
            return JSONResponse(
                status_code=400,
                content={"error": "Файл должен быть в формате CSV"}
            )
        
        # Читаем содержимое файла
        contents = await file.read()
        
        # Сохраняем файл как data.csv
        with open('data.csv', 'wb') as f:
            f.write(contents)
        
        return JSONResponse(
            content={
                "message": "CSV файл успешно загружен",
                "filename": file.filename,
                "size": len(contents)
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка при обработке файла: {str(e)}"}
        )

@app.get("/download/excel")
async def download_excel():
    """Скачать Excel файл с результатами"""
    import os
    if not os.path.exists('result.xlsx'):
        return JSONResponse(
            status_code=404,
            content={"error": "Файл result.xlsx не найден. Сначала выполните анализ."}
        )
    return FileResponse(
        path='result.xlsx',
        filename='linear_regression_results.xlsx',
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@app.get("/download/plot")
async def download_plot():
    """Скачать график matplotlib"""
    import os
    if not os.path.exists('result.png'):
        return JSONResponse(
            status_code=404,
            content={"error": "График result.png не найден. Сначала выполните анализ."}
        )
    return FileResponse(
        path='result.png',
        filename='linear_regression_plot.png',
        media_type='image/png'
    )

@app.get("/plot")
async def get_plot():
    """Получить график для отображения на сайте"""
    import os
    if not os.path.exists('result.png'):
        return JSONResponse(
            status_code=404,
            content={"error": "График не найден. Сначала выполните анализ."}
        )
    return FileResponse(
        path='result.png',
        media_type='image/png'
    )


