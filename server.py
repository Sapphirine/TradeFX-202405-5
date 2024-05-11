import json
from flask import Flask
from flask import render_template
from flask import request, jsonify
import pandas as pd
import plotly
import warnings
import plotly.graph_objs as go
warnings.filterwarnings("ignore")
from src.ml import predict, init, train, evl
from src.advisor import get_currency_info, get_n_latest_news, client


# Initialize the Flask application
app = Flask(__name__)

app.config['TEMPLATES_AUTO_RELOAD'] = True


# Define routes and their corresponding view functions
@app.route('/')
def main():
    return render_template('home.html')   # homepage


@app.route('/currency_info/', methods=['GET'])
def currency_info():
    currency = request.args.get('currency') # Get currency from query parameters

    # for now the time period is hardcoded
    start_date = '2019-05-01'
    end_date = '2020-04-30'

    # Retrieve and calculate currency information based on input parameters
    currency_info = get_currency_info(input_csv='./data/sentiment_output.csv', currency=currency, start_date=start_date, end_date=end_date)

    # Get USD info for comparison
    USD_info = get_currency_info(input_csv='./data/sentiment_output.csv', currency='USD', start_date=start_date, end_date=end_date)

    # Set the risk threshold
    threshold = 1.5

    if abs(currency_info['avg']-USD_info['avg'])>threshold:
        risk = "High"
    else:
        risk = "Low"
    result = {'currency':currency,'start':currency_info['start'],'end':currency_info['end'],"sum":currency_info['sum'],"avg":currency_info['avg'],"risk":risk}

    return jsonify(result) # Return result as JSON

@app.route('/askAI/', methods=['GET'])
def askAI():
    # Retrieve query parameters
    headline = request.args.get('h')
    brief = request.args.get('b')
    currency = request.args.get('c')
    prompt = request.args.get('prompt')

    # Call to OpenAI API to get AI-generated completion based on the currency news
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a foreign exchange trader in the foreign exchange and the finanical fields."},
        {"role": "user", "content": f"Based on the following headline and brief '{headline}\n{brief}', provide a detailed analysis for the currency {currency}. Analyze the question below: {prompt}. Return the answer in html format instead of markdown."},
    ]
    )
    # Remove markdown formatting and return the HTML content
    return completion.choices[0].message.content.replace("```html","").replace("```","")


@app.route('/advise/', methods=['GET'])
def advise():
    # Retrieve query parameters
    prompt = request.args.get('prompt')
    # Similar to askAI, but simpler request handling
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a foreign exchange trader in the foreign exchange and the finanical fields."},
        {"role": "user", "content": f"Answer the question: {prompt}."},
    ]
    )
    return completion.choices[0].message.content.replace("```html","").replace("```","")

@app.route('/news/', methods=['GET'])
def ten_news():
    currency = request.args.get('currency')
    # Retrieve the latest 10 news articles for the given currency
    news = get_n_latest_news(10, currency, input_csv='./data/news_example.csv')
    return jsonify(news)


@app.route('/model/')
def model():
    currency_pair = request.args.get('currencyPair')
    predictions_df = predict(currency_pair)  # Generate predictions for the currency pair

    # Plot the predictions using Plotly
    fig = go.Figure()
    x_values = predictions_df.index.strftime('%Y-%m-%d %H:%M:%S')  # Format date with hour
    fig.add_trace(go.Scatter(x=x_values, y=predictions_df['Predictions'], mode='lines', name='Predictions'))
    fig.update_layout(xaxis_title='Time', yaxis_title='Prediction')

    # Convert Plotly figure to JSON for rendering in the frontend
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('model.html', currency_pair=currency_pair, plot_json=plot_json)


@app.route('/trade/')
def trade():
    currency_pair = request.args.get('currencyPair')
    trade_type = request.args.get('T')  # Get the type of trade from request, long term or short term
    initial_value = request.args.get('v', type=float, default=10000)   # Get initial investment value, defaulting to $10,000
    if trade_type == 'L':
        window_size = 30*24
    elif trade_type == 'S':
        window_size = 24

    # Read and prepare data for trading simulation
    data = pd.read_csv('./data/predictions_combined_'+currency_pair+'.csv')
    new_df = pd.DataFrame()
    new_df['Timestamp'] = data.index
    new_df['Close'] = data['Predictions']

    model = init(currency_pair,initial_value,window_size,new_df)
    model = train(model,window_size)

    evl_data = pd.read_csv('./data/predictions_' + currency_pair + '.csv')
    evl_df = pd.DataFrame()
    evl_df['Timestamp'] = evl_data.index
    evl_df['Close'] = evl_data['Predictions']

    temp_filename, metrics = evl(initial_value,window_size,evl_df,window_size,model)

    # Serve the trading page with the simulation results and plot
    return render_template('trade.html', initial_value = initial_value, type = trade_type, profit = metrics['profit'],plot_path="../static/"+temp_filename)

if __name__ == '__main__':
    app.run(debug=True)  # Start the Flask application with debug enabled
