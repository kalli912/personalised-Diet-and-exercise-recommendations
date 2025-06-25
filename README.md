# Precision Medicine - Diet & Exercise Recommendations

A Streamlit-based web application that provides personalized diet and exercise recommendations based on individual health profiles using machine learning and data analysis.

## 🚀 Features

- **Personalized Health Dashboard**: BMI analysis, sleep assessment, and health metrics visualization
- **Exercise Recommendations**: Activity suggestions based on BMI category and fitness goals
- **Diet Recommendations**: Food suggestions tailored to individual health profiles
- **Health Risk Assessment**: Risk evaluation with personalized recommendations
- **Interactive Visualizations**: BMI gauge charts and health metrics displays

## 📊 Data Sources

The application uses several health and nutrition datasets:

- **Health & Fitness Dataset**: 687,701 records of health and fitness data (compressed to 10MB)
- **Food Database**: Nutritional information for various food items
- **Sleep & Lifestyle Dataset**: Sleep patterns and lifestyle factors

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd precision-medicine
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
precision-medicine/
├── app.py                          # Main Streamlit application
├── model_builder.py                # Model training and data processing
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── data/                           # Data files
│   ├── health_fitness_dataset_compressed.csv.bz2  # Compressed health data (10MB)
│   ├── FOOD-DATA-GROUP1.csv        # Food nutritional data
│   ├── Sleep_health_and_lifestyle_dataset.csv     # Sleep data
│   └── Combined_FOOD_METADATA.csv  # Food metadata
├── models/                         # Model files
│   └── __init__.py
└── venv/                          # Virtual environment (excluded from git)
```

## 🎯 Key Optimizations

- **Data Compression**: Health dataset compressed from 75MB to 10MB using bz2 compression
- **Memory Optimization**: Data types optimized to reduce memory usage by 87%
- **Git-Friendly**: Large files excluded, repository size reduced to ~10MB
- **No Model Dependencies**: Application loads data directly, no need for large pickle files

## 🔧 Technical Details

- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Machine Learning**: Scikit-learn (for data preprocessing)
- **Compression**: bz2 for optimal compression ratio

## 📈 Performance

- **Compression Ratio**: 86.5% reduction in dataset size
- **Memory Usage**: 87.3% reduction in memory footprint
- **Data Integrity**: 100% preserved during compression

## 🚀 Usage

1. Fill out your personal information in the sidebar
2. Click "Get Recommendations" to receive personalized advice
3. Explore different tabs for comprehensive health insights:
   - Health Dashboard
   - Exercise Recommendations
   - Diet Recommendations
   - Health Risk Assessment

## 📝 Notes

- The application uses rule-based recommendations instead of pre-trained models for simplicity
- All data is loaded from compressed files for efficient storage and transfer
- The virtual environment is excluded from git to keep repository size minimal

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the application
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License. 