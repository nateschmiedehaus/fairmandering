# Fairmandering - A Fair Redistricting System

Fairmandering is an advanced, comprehensive redistricting system designed to create fair and legally compliant districting plans. It utilizes cutting-edge data processing techniques, multi-objective optimization algorithms, and interactive visualizations to ensure transparency, robustness, and usability. This project is ideal for understanding and implementing fair districting practices using state-of-the-art data science and software development tools.

## 🚀 Features

- **Data Integration**: Integrates data from multiple sources, including Census, FEC, BLS, HUD, and EPA.
- **Trend Analysis**: Projects future demographic and voting patterns based on historical data.
- **Multi-Objective Optimization**: Utilizes NSGA-III to balance various fairness criteria like population equality, minority representation, and competitiveness.
- **Ensemble Analysis**: Generates and evaluates multiple redistricting plans to identify the best solutions.
- **Sensitivity Analysis**: Assesses the robustness of solutions to parameter variations.
- **Interactive User Interface**: Combines a web-based interface (using Tailwind CSS and D3.js) with command-line capabilities for flexible usage.
- **Visualization**: Interactive and visually engaging maps, Tableau dashboards, and dynamic Plotly charts.
- **Plan Versioning**: A robust system for versioning redistricting plans, ensuring traceability and transparency.
- **Security**: Encryption and secure key management for sensitive data, ensuring data privacy and protection.
- **Comprehensive Documentation**: Includes API references, user guides, and detailed examples for ease of use.

## 🏗️ Setup

### Prerequisites
Ensure you have Python 3.8+ and all required libraries installed.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nateschmiedehaus/fairmandering.git
   cd fairmandering
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   - Copy `.env.example` to `.env` and fill in the necessary values:
   ```text
   TABLEAU_SERVER_URL=https://public.tableau.com/
   FLASK_SECRET_KEY=your_secret_key
   ```

4. **Build the Tailwind CSS**:
   ```bash
   npx tailwindcss -i ./static/css/input.css -o ./static/css/output.css --watch
   ```

## 💻 Usage

1. **Start the Flask app**:
   ```bash
   python main.py
   ```
   
2. **Access the application**:
   - Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).
   
3. **Interacting with the application**:
   - Use the web interface to enter a state FIPS code and run the redistricting process.
   - View and download interactive reports and visualizations, including Tableau dashboards and Plotly charts.

## 🧪 Testing

To run the tests:
```bash
pytest
```

Make sure tests cover routes, API integrations, and data processing. You can expand test cases in the `tests/` directory to cover more scenarios.

## 📈 Visualizations

- **Tableau Dashboards**: Embedded Tableau visualizations provide dynamic insights into districting plans.
- **Plotly Charts**: Interactive fairness metrics and demographic distributions.
- **D3.js**: Custom visualizations for trend analysis and comparative analysis.
- **Folium Maps**: Interactive district maps for geographic analysis.

## 🔒 Security

Fairmandering includes encryption for sensitive data and secure management of API keys through environment variables. Ensure all sensitive information is stored securely in `.env` and avoid hardcoding keys.

## 📚 Documentation

- API References: Detailed information on available APIs.
- User Guides: Step-by-step instructions for setting up and using the application.
- Examples: Sample input and output files for understanding system behavior.

## 🤝 Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact

For further information, reach out to Nathaniel Schmiedehaus at nate@schmiedehaus.com.
