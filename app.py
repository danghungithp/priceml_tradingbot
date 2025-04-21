# Thêm eventlet và monkey_patch ngay từ đầu
import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import plotly.io as pio
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression # Import mô hình ML ví dụ
from sklearn.model_selection import train_test_split # Để tách dữ liệu (tùy chọn)
import warnings

# Tắt các cảnh báo không cần thiết từ sklearn/pandas
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key!' # Thay bằng một khóa bí mật thực tế
# async_mode='eventlet' là quan trọng để tương thích
socketio = SocketIO(app, async_mode='eventlet')
scheduler = BackgroundScheduler(daemon=True) # Chạy nền và tự thoát khi app chính thoát

# Biến toàn cục để lưu trữ dữ liệu mới nhất
latest_chart_json = None
latest_signals = []

# --- Cấu hình và Khởi tạo ---
# (Thêm các cấu hình cần thiết ở đây)

# --- Lấy và Xử lý Dữ liệu ---
def get_btc_data(interval='15m', period='60d'):
    """
    Lấy dữ liệu BTC/USD từ Yahoo Finance.
    Mặc định lấy khung 15 phút ('15m') trong 60 ngày ('60d') gần nhất.
    Yahoo Finance thường giới hạn dữ liệu intraday (như 15m) trong 60 ngày.
    Để lấy dữ liệu 1 năm, cần dùng interval='1d'.
    """
    print(f"Đang tải dữ liệu BTC-USD, interval={interval}, period={period}...")
    try:
        # Sử dụng period thay vì start/end date cho dữ liệu intraday gần đây
        data = yf.download(tickers='BTC-USD', period=period, interval=interval)

        if data.empty:
            print(f"Không tải được dữ liệu {interval} từ yfinance cho {period} gần nhất.")
            # Thử lấy dữ liệu hàng ngày nếu 15m thất bại hoặc không có
            if interval != '1d':
                print("Thử tải dữ liệu hàng ngày ('1d') cho 1 năm ('1y')...")
                data = yf.download(tickers='BTC-USD', period='1y', interval='1d')
                if data.empty:
                    print("Không tải được cả dữ liệu hàng ngày.")
                    return None
                else:
                     print("Đã tải dữ liệu hàng ngày.")
            else:
                 return None # Nếu lấy 1d cũng thất bại

        # Chuyển đổi index thành cột Datetime và chuẩn hóa múi giờ (nếu có)
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        if data.index.tz is not None:
             # Chuyển về UTC rồi loại bỏ thông tin múi giờ để đơn giản hóa
             data.index = data.index.tz_convert('UTC').tz_localize(None)
        else:
             # Gán múi giờ là UTC nếu không có, rồi loại bỏ
             data.index = data.index.tz_localize('UTC').tz_localize(None)


        # Đổi tên cột cho dễ sử dụng (yf có thể trả về chữ thường hoặc tuple)
        new_columns = []
        for col in data.columns:
            if isinstance(col, str):
                new_columns.append(col.capitalize())
            elif isinstance(col, tuple): # Xử lý trường hợp cột là tuple (ví dụ: MultiIndex)
                # Cố gắng lấy tên chính hoặc nối các phần tử tuple
                name = col[0] if col[0] else '_'.join(filter(None, col))
                new_columns.append(name.capitalize() if isinstance(name, str) else str(name))
            else:
                new_columns.append(str(col)) # Chuyển thành chuỗi nếu không phải str/tuple
        data.columns = new_columns

        # Đổi tên 'Adj close' thành 'Adj_close' nếu tồn tại để tránh lỗi sau này
        if 'Adj close' in data.columns:
             data = data.rename(columns={'Adj close': 'Adj_close'})

        # Đảm bảo có các cột cần thiết sau khi đổi tên
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
             print(f"Cảnh báo: Dữ liệu tải về có thể thiếu các cột cần thiết sau khi xử lý: {required_cols}. Các cột hiện có: {list(data.columns)}")
             # Không trả về None ngay, cố gắng tiếp tục nếu có thể

        print(f"Dữ liệu tải về: {len(data)} dòng, từ {data.index.min()} đến {data.index.max()} UTC")
        return data

    except Exception as e:
        print(f"Lỗi khi tải dữ liệu yfinance: {e}")
        return None

# --- Feature Engineering ---
def calculate_features(df, sma_periods=[10, 20], rsi_period=14):
    """Tính toán các đặc trưng kỹ thuật (SMA, RSI)."""
    if df is None or df.empty:
        return df

    df_feat = df.copy() # Tạo bản sao để tránh thay đổi df gốc

    # Simple Moving Averages (SMA)
    for period in sma_periods:
        df_feat[f'SMA_{period}'] = df_feat['Close'].rolling(window=period).mean()

    # Relative Strength Index (RSI)
    delta = df_feat['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()

    rs = avg_gain / avg_loss
    df_feat[f'RSI_{rsi_period}'] = 100 - (100 / (1 + rs))

    # Xóa các hàng có giá trị NaN do tính toán rolling window
    df_feat.dropna(inplace=True)
    return df_feat


# --- Logic Price Action & Machine Learning ---

def price_action_signals(df):
    """Xác định tín hiệu dựa trên các mẫu hình Price Action đơn giản."""
    signals = []
    if df is None or len(df) < 2:
        return signals

    # Ví dụ: Phát hiện mẫu hình Nến Nhấn Chìm (Engulfing) đơn giản
    for i in range(1, len(df)):
        prev_row = df.iloc[i-1]
        curr_row = df.iloc[i]

        # Bullish Engulfing (Nến xanh nhấn chìm nến đỏ trước đó)
        is_prev_red = prev_row['Open'] > prev_row['Close']
        is_curr_green = curr_row['Close'] > curr_row['Open']
        is_engulfing_bullish = is_prev_red and is_curr_green and \
                               curr_row['Close'] > prev_row['Open'] and \
                               curr_row['Open'] < prev_row['Close']

        # Bearish Engulfing (Nến đỏ nhấn chìm nến xanh trước đó)
        is_prev_green = prev_row['Close'] > prev_row['Open']
        is_curr_red = curr_row['Open'] > curr_row['Close']
        is_engulfing_bearish = is_prev_green and is_curr_red and \
                               curr_row['Open'] > prev_row['Close'] and \
                               curr_row['Close'] < prev_row['Open']

        if is_engulfing_bullish:
            signals.append({
                'time': curr_row.name.strftime('%Y-%m-%d %H:%M:%S'), # Sử dụng index làm thời gian
                'type': 'BUY',
                'price': curr_row['Close'],
                'strategy': 'PA: Bullish Engulfing'
            })
        elif is_engulfing_bearish:
             signals.append({
                'time': curr_row.name.strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'SELL',
                'price': curr_row['Close'],
                'strategy': 'PA: Bearish Engulfing'
            })
    # Chỉ giữ lại các tín hiệu gần đây nhất (ví dụ: 10 tín hiệu)
    return signals[-10:]

def machine_learning_signals(df_original):
    """Tạo tín hiệu bằng mô hình Logistic Regression đơn giản (ví dụ)."""
    signals = []
    if df_original is None or len(df_original) < 50: # Cần đủ dữ liệu để tính features và huấn luyện
        print("Không đủ dữ liệu để tạo tín hiệu ML.")
        return signals

    # 1. Feature Engineering
    df_feat = calculate_features(df_original)
    if df_feat.empty:
        print("Không thể tính toán features cho ML.")
        return signals

    # 2. Chuẩn bị dữ liệu cho mô hình
    # Đặc trưng (X): SMA, RSI
    features = [col for col in df_feat.columns if 'SMA_' in col or 'RSI_' in col]
    X = df_feat[features]

    # Mục tiêu (y): Dự đoán giá sẽ tăng (1) hay giảm (0) trong kỳ tiếp theo
    # Đây là một cách định nghĩa mục tiêu đơn giản, có thể cải thiện
    df_feat['Target'] = (df_feat['Close'].shift(-1) > df_feat['Close']).astype(int)
    df_feat.dropna(inplace=True) # Xóa hàng cuối cùng không có target

    y = df_feat['Target']
    X = X.loc[y.index] # Đảm bảo X và y cùng index sau khi dropna

    if len(X) < 20: # Cần đủ dữ liệu sau khi tính feature và target
        print("Không đủ dữ liệu sau khi chuẩn bị cho ML.")
        return signals

    # Tách dữ liệu (ví dụ: 80% train, 20% test - không bắt buộc cho ví dụ này)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 3. Huấn luyện mô hình Logistic Regression đơn giản
    # Huấn luyện trên toàn bộ dữ liệu có sẵn cho ví dụ này
    model = LogisticRegression(solver='liblinear', random_state=42)
    try:
        model.fit(X, y)
    except ValueError as e:
        print(f"Lỗi khi huấn luyện mô hình ML: {e}")
        return signals


    # 4. Dự đoán cho điểm dữ liệu cuối cùng
    last_features = X.iloc[-1:] # Lấy hàng cuối cùng của features
    try:
        prediction = model.predict(last_features)[0]
        probability = model.predict_proba(last_features)[0] # Xác suất dự đoán
    except Exception as e:
         print(f"Lỗi khi dự đoán bằng mô hình ML: {e}")
         return signals

    # 5. Tạo tín hiệu dựa trên dự đoán
    signal_type = 'BUY' if prediction == 1 else 'SELL'
    confidence = probability[prediction] # Lấy xác suất của lớp được dự đoán

    # Chỉ tạo tín hiệu nếu mô hình "khá chắc chắn" (ví dụ: xác suất > 0.6)
    if confidence > 0.6:
        signals.append({
            'time': last_features.index[0].strftime('%Y-%m-%d %H:%M:%S'), # Thời gian của điểm dữ liệu dùng để dự đoán
            'type': signal_type,
            'price': df_original['Close'].loc[last_features.index[0]], # Giá đóng cửa tại thời điểm đó
            'strategy': f'ML: LogReg (Conf: {confidence:.2f})'
        })

    return signals


# Modify generate_signals to return df as well, for consistency
def generate_signals(df_original):
    """Kết hợp tín hiệu từ Price Action và Machine Learning."""
    if df_original is None or df_original.empty:
        print("DataFrame gốc rỗng trong generate_signals.")
        return [], df_original # Trả về df gốc nếu không xử lý được

    # Tính toán tín hiệu
    # PA signals dùng df gốc
    pa_signals = price_action_signals(df_original)
    # ML signals cũng nên dùng df gốc, hàm ML sẽ tự tính feature nếu cần
    ml_signals = machine_learning_signals(df_original)

    # Kết hợp và sắp xếp tín hiệu
    all_signals = sorted(pa_signals + ml_signals, key=lambda x: x['time'], reverse=True)

    # Giới hạn số lượng tín hiệu
    limited_signals = all_signals[:15]

    # Trả về tín hiệu và DataFrame gốc (để đảm bảo biểu đồ dùng đúng index)
    print(f"Đã tạo {len(limited_signals)} tín hiệu.")
    return limited_signals, df_original


# --- Tạo Biểu đồ ---
# Modify create_candlestick_chart to accept signals_list and improve robustness
def create_candlestick_chart(df, signals_list):
    """Tạo biểu đồ nến bằng Plotly và hiển thị tín hiệu."""
    if df is None or df.empty:
        print("DataFrame rỗng, không thể tạo biểu đồ.")
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            print("Đã chuyển đổi index thành DatetimeIndex.")
        except Exception as e:
            print(f"Lỗi chuyển đổi index thành DatetimeIndex: {e}. Không thể tạo biểu đồ.")
            return None

    print(f"Tạo biểu đồ với {len(df)} điểm dữ liệu và {len(signals_list)} tín hiệu đầu vào.")

    fig = go.Figure() # Khởi tạo Figure trống

    # 1. Thêm trace Candlestick
    try:
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name='BTC/USD'))
        print("Đã thêm trace Candlestick.")
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi thêm trace Candlestick: {e}")
        # Nếu không vẽ được nến thì không nên tiếp tục
        return None

    # 2. Thêm các trace tín hiệu (Scatter)
    signal_configs = {
        'PA Buy': {'filter': lambda s: s['type'] == 'BUY' and s['strategy'].startswith('PA:'), 'marker': dict(color='lime', size=11, symbol='triangle-up')},
        'PA Sell': {'filter': lambda s: s['type'] == 'SELL' and s['strategy'].startswith('PA:'), 'marker': dict(color='red', size=11, symbol='triangle-down')},
        'ML Buy': {'filter': lambda s: s['type'] == 'BUY' and s['strategy'].startswith('ML:'), 'marker': dict(color='cyan', size=9, symbol='circle')},
        'ML Sell': {'filter': lambda s: s['type'] == 'SELL' and s['strategy'].startswith('ML:'), 'marker': dict(color='magenta', size=9, symbol='circle')}
    }

    for name, config in signal_configs.items():
        # Lọc tín hiệu theo cấu hình
        filtered_signals = [s for s in signals_list if config['filter'](s)]
        print(f"Tổng số tín hiệu '{name}' trước khi lọc thời gian: {len(filtered_signals)}")

        if not filtered_signals:
            continue

        # Chuyển đổi thời gian và lọc theo index của df
        signal_times = pd.to_datetime([s['time'] for s in filtered_signals])
        valid_indices = signal_times.isin(df.index)
        valid_times = signal_times[valid_indices]
        valid_signals = [s for s, is_valid in zip(filtered_signals, valid_indices) if is_valid]

        print(f"Số tín hiệu '{name}' hợp lệ (thời gian khớp index): {len(valid_signals)}")

        if valid_signals:
            try:
                fig.add_trace(go.Scatter(
                    x=valid_times,
                    y=[s['price'] for s in valid_signals],
                    mode='markers',
                    marker=config['marker'],
                    name=name,
                    hoverinfo='text',
                    text=[f"{name}<br>Price: {s['price']:.2f}<br>Time: {s['time']}<br>Strategy: {s.get('strategy', '')}" for s in valid_signals]
                ))
                print(f"Đã thêm trace Scatter cho '{name}'.")
            except Exception as e:
                 print(f"Lỗi khi thêm trace Scatter cho '{name}': {e}")
                 # Có thể bỏ qua lỗi này và tiếp tục vẽ các tín hiệu khác

    # 3. Cập nhật layout
    fig.update_layout(
        title='Biểu đồ giá BTC/USD và Tín hiệu Giao dịch',
        xaxis_title='Thời gian (UTC)',
        yaxis_title='Giá (USD)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        legend_title_text='Chú giải',
        # Đảm bảo layout đủ rộng để hiển thị
        autosize=True,
        margin=dict(l=50, r=50, b=100, t=100, pad=4) # Điều chỉnh margin nếu cần
    )

    try:
        chart_json = pio.to_json(fig)
        print("Đã tạo JSON cho biểu đồ thành công.")
        return chart_json
    except Exception as e:
        print(f"Lỗi khi chuyển biểu đồ thành JSON: {e}")
        return None

# --- Routes ---
@app.route('/')
def index():
    """Trang chính hiển thị biểu đồ và tín hiệu."""
    print("\n--- Yêu cầu mới đến route / ---")
    btc_data_original = get_btc_data(interval='15m', period='60d')
    signals = []
    chart_json = None
    chart_df = None # DataFrame dùng để vẽ biểu đồ

    if btc_data_original is not None and not btc_data_original.empty:
        # Tạo tín hiệu và lấy DataFrame gốc để vẽ biểu đồ
        signals, chart_df = generate_signals(btc_data_original)

        # Tạo biểu đồ từ DataFrame gốc và tín hiệu đã tạo
        if chart_df is not None and not chart_df.empty:
             print(f"Gọi create_candlestick_chart với df {len(chart_df)} dòng.")
             chart_json = create_candlestick_chart(chart_df, signals) # Truyền tín hiệu vào
        else:
             print("Không có DataFrame hợp lệ để tạo biểu đồ.")
    else:
        print("Không thể tải dữ liệu BTC hoặc dữ liệu rỗng.")

    current_year = datetime.now().year
    print(f"Render template ban đầu. Dữ liệu sẽ được gửi qua SocketIO.")
    # Chỉ cần render template, dữ liệu sẽ được gửi qua socket
    return render_template('index.html', now={'year': current_year})

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    """Gửi dữ liệu mới nhất cho client vừa kết nối."""
    print('Client đã kết nối')
    if latest_chart_json and latest_signals:
        print('Gửi dữ liệu hiện có cho client mới')
        emit('update_data', {'chart': latest_chart_json, 'signals': latest_signals})
    else:
        # Nếu chưa có dữ liệu, có thể kích hoạt cập nhật lần đầu
        print('Chưa có dữ liệu, sẽ cập nhật sớm.')
        # Hoặc gọi update_data_and_emit() nếu muốn cập nhật ngay
        # update_data_and_emit() # Cẩn thận nếu hàm này tốn thời gian

@socketio.on('disconnect')
def test_disconnect():
    print('Client đã ngắt kết nối')


# --- Background Task ---
def update_data_and_emit():
    """Lấy dữ liệu mới, tạo biểu đồ/tín hiệu và gửi qua SocketIO."""
    global latest_chart_json, latest_signals
    print("\n--- [Background Task] Bắt đầu cập nhật dữ liệu ---")
    # Lấy dữ liệu 15 phút gần nhất (ví dụ: 7 ngày để đủ cho tính toán)
    # Điều chỉnh period nếu cần nhiều dữ liệu hơn cho ML/PA
    btc_data_original = get_btc_data(interval='15m', period='7d')
    signals = []
    chart_json = None
    chart_df = None

    if btc_data_original is not None and not btc_data_original.empty:
        signals, chart_df = generate_signals(btc_data_original)
        if chart_df is not None and not chart_df.empty:
            print(f"Gọi create_candlestick_chart với df {len(chart_df)} dòng.")
            chart_json = create_candlestick_chart(chart_df, signals)
        else:
            print("[Background Task] Không có DataFrame hợp lệ để tạo biểu đồ.")
    else:
        print("[Background Task] Không thể tải dữ liệu BTC hoặc dữ liệu rỗng.")

    if chart_json and signals:
        print("[Background Task] Dữ liệu mới đã sẵn sàng. Gửi qua SocketIO.")
        print(f"[Background Task] Chart JSON (đầu): {chart_json[:200]}...") # In 200 ký tự đầu để kiểm tra
        latest_chart_json = chart_json
        latest_signals = signals
        # Gửi dữ liệu mới đến tất cả các client đang kết nối
        socketio.emit('update_data', {'chart': latest_chart_json, 'signals': latest_signals})
    else:
        print("[Background Task] Không tạo được dữ liệu mới để gửi.")
    print("--- [Background Task] Kết thúc cập nhật ---")


@app.route('/subscribe', methods=['POST'])
def subscribe():
    """Xử lý việc đăng ký nhận tín hiệu qua email."""
    email = request.form.get('email')
    if email:
        # Logic xử lý lưu email hoặc gửi email chào mừng (sẽ thêm sau)
        print(f"Email đăng ký: {email}")
        # Phản hồi thành công (có thể dùng flash message của Flask)
        return jsonify({'status': 'success', 'message': 'Đăng ký thành công!'})
    else:
        return jsonify({'status': 'error', 'message': 'Vui lòng nhập email hợp lệ.'}), 400

# --- Chạy ứng dụng ---
if __name__ == '__main__':
    # Lên lịch chạy hàm update_data_and_emit mỗi 15 phút
    # Chạy lần đầu ngay lập tức (hoặc sau vài giây)
    scheduler.add_job(update_data_and_emit, 'interval', minutes=15, id='btc_update_job', replace_existing=True, misfire_grace_time=60)
    scheduler.start()
    print("Scheduler đã bắt đầu. Chạy cập nhật lần đầu...")
    # Chạy lần đầu để có dữ liệu ngay khi khởi động
    update_data_and_emit()

    print("Khởi chạy ứng dụng Flask-SocketIO với eventlet...")
    # Sử dụng socketio.run thay vì app.run
    # host='0.0.0.0' để có thể truy cập từ bên ngoài container (khi chạy cục bộ)
    # port=7860 là cổng thường dùng trên Hugging Face Spaces
    # debug=False cho môi trường giống production hơn
    # use_reloader=False vì Gunicorn sẽ quản lý process
    # Lưu ý: Khi chạy qua Dockerfile/Gunicorn, dòng này không được thực thi trực tiếp.
    socketio.run(app, host='0.0.0.0', port=7860, debug=False, use_reloader=False)
