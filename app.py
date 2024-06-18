from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd
import random
from werkzeug.utils import secure_filename
import os, glob, numpy as np
from pytube import Playlist
import os
import moviepy.editor as mp
from bs4 import BeautifulSoup as bs
from selenium import webdriver
import time

# render_template --> html 파일로 링크 연결해주는 기능
# 참고링크
# https://medium.com/@mystar09070907/%EC%9B%B9-%ED%8E%98%EC%9D%B4%EC%A7%80-client-%EC%97%90%EC%84%9C-%EC%A0%95%EB%B3%B4-%EB%B3%B4%EB%82%B4%EA%B8%B0-bf3aff952d3d


app = Flask(__name__)


# __name__은 파이썬이 내부적으로 사용하는 특별한 변수명

@app.route('/')
def intro():
    return render_template('Main.html')


@app.route('/about')
def about():
    return render_template('About(menu1).html')


# ----------------------------------가사빈도수 노래추천----------------------------------------------------
@app.route('/lyrics')
def lyrics():
    return render_template('lyrics_sim.html')


# GET방식 (DB에서 찾아서 조회(가져오는것), 주소줄에 값인남음
# POST방식 (DB에 새로운 정보 저장(수행하는것), 숨겨져서 body에보내짐)


# 가사유사도분석 유사노래추천 top10
@app.route("/post", methods=['POST'])
def post():
    filepath = 'static/database/SongChorus.csv'
    df = pd.read_csv(filepath)
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop_duplicates(keep='first', ignore_index=True)
    count_vect = CountVectorizer(min_df=0, ngram_range=(1, 3)).fit(df['가사'])
    count_vect.vocabulary_
    lyrics_matrix = count_vect.fit_transform(df['가사'])
    lyrics_similarity = cosine_similarity(lyrics_matrix, lyrics_matrix)
    lyrics_similarity_sorted_idx = lyrics_similarity.argsort()[:, ::-1]

    song_name = request.form['text']
    # 템플릿의 input값을 가져옴

    title_song = df[df['제목'] == song_name]
    title_song = title_song
    song_index = title_song.index.values
    iloc = lyrics_similarity_sorted_idx[song_index, :11]
    quiz = iloc.reshape(-1)
    quiz = quiz[quiz != song_index]
    df = df.iloc[quiz][:11]

    # # 방법1)df를 to_html()로 바로 랜더링
    # return render_template('lyrics_sim_result.html', tables=[df.to_html(classes='data', header="true")], song_name=song_name)

    # 방법2)df를 그대로 받아서 table 태그로 랜더링
    return render_template('lyrics_sim_result.html',song_name=song_name,
                           column_names=df.columns.values,
                           row_data=list(df.values.tolist()),
                           zip=zip)


# flask에서 데이터프레임은 그냥 보이지 않기 때문에 to.html을 써 주어야함
# 템플릿에 데이터프레임 형태 붙이기 참고
# https://stackoverflow.com/questions/52644035/how-to-show-a-pandas-dataframe-into-a-existing-flask-html-table


# ----------------------------------가사 정서 유사도 노래추천----------------------------------------------------
@app.route('/lyrics_emotion')
def lyrics_emotion():
    return render_template('lyrics_emotion_sim.html')



# ----------------------------------일기장 정서분석 노래추천-----------------------------------------------------

# 일기장 정서분석 노래추천 top10

@app.route('/diary')
def diary():
    return render_template('diary.html')


@app.route("/diary_post", methods=['POST'])
def diary_result():
    return render_template('diary_result.html',print_emotion = diary_post(),  print_text1=diary_post(),
                           print_text2=diary_post())  # diary_post()함수의 실행결과 return값을 같은 변수명에 담는다


def diary_post():
    selected_words = pd.read_csv('static/database/selected_words.csv', encoding='cp949')
    selected_words = selected_words.columns

    # all_docs[0]
    all_words = pd.read_csv('static/database/all_words.csv', encoding='cp949')
    all_words = all_words.columns
    # all_docs[0]
    selected_words = selected_words.tolist()
    all_words = all_words.tolist()
    a_lst = []

    for a in all_words:
        try:
            a_lst.append(eval(a))
        except:
            a_lst.append(eval(a[:-2]))

    a_lst

    ####################################
    filepath = 'static/database/After_Data_Pretreatment_Processing(13486).csv'
    df_song = pd.read_csv(filepath)

    ####################################
    loaded_model = joblib.load('static/database/Extraction_Emotion.joblib')

    okt = Okt()

    emotion_number_list = ["행복", "놀람", "분노", "공포", "혐오", "슬픔", "중립"]
    emotion_result_list = ["1", "2", "3", "4", "5", "6", "7"]

    def term_frequency(doc):
        return [doc.count(word) for word in selected_words]

    # all_docs_train = [term_frequency(d) for d, _ in a_lst]
    # all_docs_test = [c for _, c in a_lst]

    def tokenize(doc):
        # norm은 정규화, stem은 근어로 표시하기를 나타냄
        return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

    ####################################
    sadlist = ['(슬픔)  힘들 땐 무조건 내가 제일 힘든 것이다. 그건 이기적인 게 아니다. -베스트셀러 죽고 싶지만 떡볶이는 먹고 싶어'], [
        '(슬픔)  삶은 실수투성이야, 우리는 늘 실수를 하지.-영화 주토피아'], ['(슬픔)  흔들려도 좋으니 꺾이지만 마라- 동그라미 -새삼스러운 생각'], [
                  '(슬픔)  비록 아무도 과거로 돌아가 새 출발할 순 없지만, 누구나 지금 시작해 새 엔딩을 만들 수 있다. -칼 바드'], [
                  '(슬픔)  달빛에 비치는 바다도 흔들릴 때 더 아름답다. 찰랑일 때 더 빛난다 그대도 눈부시다.-흔글, 무너지지만 말아'], [
                  '(슬픔) 인간사에는 안정된 것이 하나도 없음을 기억하라. 그러므로 성공에 들뜨거나 역경에 지나치게 의기소침하지 마라 -소크라테스'], [
                  '(슬픔) 괴로운 슬픔으로 상처를 입고슬픔에 마음이 흔들릴때, 음악은 아름다운 화음으로 빠르게 치유의 손길을 건낸다.-토머스 에디슨'], [
                  '(슬픔) 절대 고개를 숙이지 말라. 고개를 쳐들고 세상을 똑바로 봐라.-헬렌 켈러'], [
                  '(슬픔) 당신은 움츠리기보다 활짝 피어나도록 만들어진 존재입니다. -오프라 원프리'], [
                  '(슬픔) 원래 좋고 나쁜 것은 다 생각하기 나름이다.-윌리엄 셰익스피어']
    happylist = ['(행복) 행복은 나누면 2배입니다'], ['(행복) 행복해진 이유를 기억하세요'], ['(행복) 역시 행복은 가까이 있어요'], [
        '(행복) 주변 사람에게도 행복을 나눠보는건 어떨까요?'], ['(행복) 행복이란 결심이요 결정이다 -앤드류 매튜스-'], [
                    '(행복) 대부분의 사람은 마음 먹은 만큼 행복하다 - 에이브러햄 링컨-']
    suprisedlist = ['복식 호흡하기. 먼저 5초 동안 크게 숨을 들이켜 복부를 팽창시킨다. 다음 5초 동안은 숨을 참고 다시 5초 동안 숨을 천천히 내쉬도록 한다.'], [
        '주변 환경과 육체의 감각에 집중하기. 자신과 주변 환경에 집중할 수 있는 마음챙김을 연습해보도록 한다. 귀에 들리는 소리, 피부로 느껴지는 온도, 냄새와 촉감을 심호흡을 하면서 집중해보자. 마음의 긴장이 풀릴 때까지 계속해서 자신과 주변 환경에 집중하도록 한다.'], [
                       '점진적 근육 이완법 써보기. 머리서부터 발끝까지 각 몸의 근육군을 차례대로 긴장시킨 뒤 이완하는 방법이다. 먼저 안면 근육에 집중을 해서 6초 동안 긴장을 주어보자. 그리고 다시 6초 동안 긴장을 서서히 풀어서 근육이 진정하는 것을 느껴보자. ']
    irelist = ['격하기 쉬운 사람이 받는 벌은 늘 행복 곁에 살면서도 행복을 손에 못 넣는 일이다 (보나르)'], [
        '다른 사람들을 자신이 원하는대로 만들 수 없다고 해서 노여워하지 마라. 왜냐하면 당신도 당신이 바라는 모습으로 자신을 만들 수 없기 때문이다. (토마스 아 켐피스)'], [
                  '노여움은 한때의 광기다. 노여움을 누르지 않으면 노여움이 당신을 누르고 만다 (호라티우스)'], ['노여움이 일면 그 결과를 생각하라 (공자)'], [
                  '모욕을 받고 이내 발칵하는 인간은 강도 아닌 조그마한 웅덩이에 불과하다 (톨스토이)'], ['분노는 무모함으로 시작해 후회로 끝난다 (피타고라스)'], [
                  '화가 날 때는 10가지 세어라. 화가 너무 많이 날 때는 100가지 세어라 (토머스 제퍼슨)']
    fearlist = ['겁을 내면 여우가 더 크게 보인다. -독일 속담'], [
        '겁을 먹는 것과 까닭 없이 불안하게 하는 두려움은 확실히 구별되는 것이지만, 이것들은 대부분 단지 상상력의 기능을 한때 중단시키는 능력의 결여로 보면 된다. -헤밍웨이'], [
                   '겁이 많은 개일수록 큰소리로 짓는다. -웹스터'], ['결과에 두려움을 느끼지 않는 사람은 거대한 다이아몬드와 같다.'], [
                   '고통받기를 두려워하는 자는 두려움 때문에 고통을 받는다. -프랑스 속담'], ['고통에는 한계가 있으나, 공포에는 한계가 없다. -플리니우스'], [
                   '공포가 있는 곳에는 행복이 없다. -세네카'], ['공포감은 이 세상의 어떠한 것보다도 수많은 사람들을 패배로 몰아 넣었다.'], [
                   '공포는 미신 때문에 생기며, 잔인성을 유발하기도 한다. -버트란트 러셀']
    aversionlist = ['겁을 내면 여우가 더 크게 보인다. -독일 속담'], [
        '겁을 먹는 것과 까닭 없이 불안하게 하는 두려움은 확실히 구별되는 것이지만, 이것들은 대부분 단지 상상력의 기능을 한때 중단시키는 능력의 결여로 보면 된다. -헤밍웨이'], [
                       '겁이 많은 개일수록 큰소리로 짓는다. -웹스터'], ['결과에 두려움을 느끼지 않는 사람은 거대한 다이아몬드와 같다.']

    target = request.form['diary_text']

    # 웹에서 받은 input값 form에 있는 name부분을 가져옴

    token = tokenize(target)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    emotion = int(loaded_model.predict(data))
    if emotion == 6:
        print_emotion = "슬픔"
        print_text1 = "위로의 말: ", random.choice(sadlist)
        list12 = df_song['title'][df_song['emotion'] == '슬픔'].index.to_list()
        e = [random.choice(list12) for i in range(1)]
        print_text2 = "제목",df_song['title'][df_song['emotion'] == '슬픔'][e].to_list(), '가수 :',df_song['artist'][df_song['emotion']=='슬픔'][e].to_list()
    if emotion == 1:
        print_emotion = "행복"
        print_text1 = "위로의 말: ", random.choice(happylist)
        list12 = df_song['title'][df_song['emotion'] == '행복'].index.to_list()
        e = [random.choice(list12) for i in range(1)]
        print_text2 = "제목",df_song['title'][df_song['emotion'] == '행복'][e].to_list(), '가수 :',df_song['artist'][df_song['emotion']=='행복'][e].to_list()
    if emotion == 2:
        print_emotion = "놀람"
        print_text1 = "위로의 말: ", random.choice(suprisedlist)
        list12 = df_song['title'][df_song['emotion'] == '놀람'].index.to_list()
        e = [random.choice(list12) for i in range(1)]
        print_text2 ="제목", df_song['title'][df_song['emotion'] == '놀람'][e].to_list(), '가수 :',df_song['artist'][df_song['emotion']=='놀람'][e].to_list()
    if emotion == 7:
        print_emotion = "중립"
        print_text1 = '중립적이시네요'
        list12 = df_song['title'][df_song['emotion'] == '중립'].index.to_list()
        e = [random.choice(list12) for i in range(1)]
        print_text2 ="제목", df_song['title'][df_song['emotion'] == '중립'][e].to_list(), '가수 :',df_song['artist'][df_song['emotion']=='중립'][e].to_list()
    if emotion == 3:
        print_emotion = "분노"
        print_text1 = "위로의 말: ", random.choice(irelist)
        list12 = df_song['title'][df_song['emotion'] == '분노'].index.to_list()
        e = [random.choice(list12) for i in range(1)]
        print_text2 ="제목", df_song['title'][df_song['emotion'] == '분노'][e].to_list(), '가수 :',df_song['artist'][df_song['emotion']=='분노'][e].to_list()
    if emotion == 4:
        print_emotion = "공포"
        print_text1 = "위로의 말: ", random.choice(fearlist)
        list12 = df_song['title'][df_song['emotion'] == '공포'].index.to_list()
        e = [random.choice(list12) for i in range(1)]
        print_text2 = "제목",df_song['title'][df_song['emotion'] == '공포'][e].to_list(), '가수 :',df_song['artist'][df_song['emotion']=='공포'][e].to_list()
    if emotion == 5:
        print_emotion = "혐오"
        print_text1 = "위로의 말: ", random.choice(aversionlist)
        list12 = df_song['title'][df_song['emotion'] == '혐오'].index.to_list()
        e = [random.choice(list12) for i in range(1)]
        print_text2 ="제목", df_song['title'][df_song['emotion'] == '혐오'][e].to_list(), '가수 :',df_song['artist'][df_song['emotion']=='혐오'][e].to_list()
    return print_emotion , print_text1, print_text2  # 리스트 형태의 리턴값임


# ----------------------------------youtube 플레이리스트 음원받기-----------------------------------------------------

# youtube 플레이리스트 음원받기
@app.route('/youtube_music')
def youtube():
    return render_template('youtube_music.html')


@app.route('/youtube_post', methods=['POST'])
def youtube_post():
    youtube_url = request.form['youtube_text']
    pl = Playlist(youtube_url)

    # 현재 프로젝트 안 music이라는 폴더에 저장됨
    for video in pl.videos:
        video.streams.first().download('music')  # jupyternotbook에서 사용할때는 ()안에 폴더이름 지운코드 사용할것

    # 작업디렉토리 변경
    os.chdir('music/')

    # 현재 작업디렉토리 표시
    path = os.getcwd()
    file_list = os.listdir(path)

    # 폴더 내 모든 파일 중 .3gpp파일 찾아내기
    # 파일명만 추출하는 코드 폴더길이에 따라 알맞게 movie_name[34:]숫자조정 필요!!!!!!
    song_list = []
    for i in range(0, len(file_list)):
        movie_name = path + '/' + file_list[i]
        song_target = movie_name[42:]
        if movie_name[-5:] == ".3gpp":
            song_list.append(song_target)

    # print(song_list)

    for i in range(0, len(song_list)):
        clip = mp.VideoFileClip(song_list[i])
        clip.audio.write_audiofile(song_list[i] + ".mp3")

    return render_template("youtube_result.html")


# ---------------------------------- melon top100 -----------------------------------------------------
@app.route('/top100list')
def top100list():
    return render_template('top100list.html')


@app.route('/top100_post', methods=['POST'])
def top100_post():
    year_text = request.form['top100_text']  #웹에서 받은 form-text값

    title_list = []
    artist_list = []

    def find_first_and_second_index(year):
        user_index = []
        user_year = []
        for i in str(year):
            user_year.append(i)
        # print(user_year)

        # 첫번째 index 선언 및 값 넣어주기
        # 2000년대
        if user_year[0] == '2':
            if user_year[2] == '2':
                user_index.append('1')
            if user_year[2] == '1':
                user_index.append('2')
            if user_year[2] == '0':
                user_index.append('3')
            # 1900년대
        if user_year[0] == '1':
            if user_year[2] == '9':
                user_index.append('4')
            if user_year[2] == '8':
                user_index.append('5')
            if user_year[2] == '7':
                user_index.append('6')
            if user_year[2] == '6':
                user_index.append('7')
            if user_year[2] == '5':
                user_index.append('8')

        user_index.append(str(-(int(user_year[3]) - 10)))
        if user_year == ['2', '0', '2', '0']:
            user_index = ['1', '1']
        # print('user_index: ' + str(user_index))

        return user_index

    def get_title(user_index):

        title_list.clear()
        artist_list.clear()
        url = 'https://www.melon.com/chart/index.htm'  # 멜론차트 페이지   # 접속할 웹 사이트 주소 (네이버)
        driver = webdriver.Chrome('D:\Download/chromedriver.exe')  # 크롬 드라이버로 크롬 켜기(로컬주소에따라 달라짐)
        driver.get(url)
        # 차트파인더 클릭
        driver.find_element_by_xpath('//*[@id="gnb_menu"]/ul[1]/li[1]/div/div/button/span').click()
        # 연대선택, 연도선택, 월선택, 장르선택
        # 연도차트 클릭
        driver.find_element_by_xpath('//*[@id="d_chart_search"]/div/h4[3]/a').click()
        time.sleep(0.3)
        # 연대선택 2010년 클릭
        driver.find_element_by_xpath(
            '//*[@id="d_chart_search"]/div/div/div[1]/div[1]/ul/li[' + str(user_index[0]) + ']/span/label').click()
        time.sleep(0.3)
        # 연도선택 2012년 클릭   ( 이것만 따로 계속 바꿔줘야되요)
        driver.find_element_by_xpath(
            '//*[@id="d_chart_search"]/div/div/div[2]/div[1]/ul/li[' + str(user_index[1]) + ']/span/label').click()
        time.sleep(0.3)
        # 장르선택 국내종합 클릭
        driver.find_element_by_xpath('//*[@id="d_chart_search"]/div/div/div[5]/div[1]/ul/li[1]/span/label').click()
        time.sleep(0.3)
        # 검색버튼 클릭
        driver.find_element_by_xpath('//*[@id="d_srch_form"]/div[2]/button/span/span').click()
        time.sleep(1)

        html = driver.page_source  # 드라이버 현재 페이지의 html 정보 가져오기
        soup = bs(html, 'lxml')

        title_elem = soup.select('div.ellipsis.rank01 > span > strong')
        artist_elem = soup.select('div.ellipsis.rank02 > span')

        driver.close()

        # 제목
        for i in title_elem:
            title_list.append(i.text)

        # 가수명
        for i in artist_elem:
            artist_list.append(i.text)

    get_title(find_first_and_second_index(year_text))  # 년도입력

    rank = []
    for i in range(1, 101):
        rank.append(i)

    df = pd.DataFrame({'순위': rank, '제목': title_list, "가수": artist_list})


    # 데이터프레임을 컬럼명, 데이터로 분리시킨다 이렇게 하고 랜더링할때 각각을 for문으로 출력할 수 있게 한다.
    # 데이터프레임을 .to_html()로 출력하는것 보다 표형태로 예쁘게 만들어서 디자인할 수 있다는 장점이 있다.
    return render_template('top100list_result.html', year_text=year_text,
                           column_names=df.columns.values,
                           row_data=list(df.values.tolist()),
                           zip=zip)


# ---------------------------------- 멜로디 fitch 유사도 -----------------------------------------------------
@app.route('/musicpitch')
def musicpitch():
    return render_template('musicpitch.html')



# ---------------------------------- 이미지분석 노래추천 -----------------------------------------------------

@app.route('/image1')
def image1():
    return render_template('image_model.html')

@app.route("/image_post", methods=['POST'])
def image_post():
    filepah = 'static/database/After_Data_Pretreatment_Processing(13486).csv'
    df_song = pd.read_csv(filepah)

    request_file = request.files['image']  # input의 key값[key값] 받아오기
    request_file.save('static/test/' + '' + secure_filename(request_file.filename))

    # 비교해볼 이미지 파일 넣는 곳
    caltech_dir = "static/test"
    # return '파일 업로드 성공!'

    # 불러올 이미지 크기
    image_w = 128
    image_h = 128
    pixels = image_h * image_w * 3

    X = []
    filenames = []
    files = glob.glob(caltech_dir + "/*.*")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        filenames.append(f)
        X.append(data)

    X = np.array(X)

    # 기존에 만든 모델 불러오기
    categories = ['거실', '공항내부', '빵집', '술집', '식당', '장난감가게', '주방', '컴퓨터실', '창고', '침실']
    model = load_model('static/database/model//multi_img_classification.model')

    prediction = model.predict(X)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    cnt = 0
    print_text1 = []
    # print_text2 = []
    # print_text3 = []
    # print_text4 = []
    # print_text5 = []
    # print_text6 = []
    # print_text7 = []
    # print_text8 = []
    # print_text9 = []
    # print_text10 = []
    # 이 비교는 그냥 파일들이 있으면 해당 파일과 비교. 카테고리와 함께 비교해서 진행하는 것은 _4 파일.
    for i in prediction:
        pre_ans = i.argmax()  # 예측 레이블
        #     print(i)
        #     print(pre_ans)
        pre_ans_str = ''
        pre_ans_str = categories[pre_ans]
        if pre_ans_str == '장난감가게':
            df_result = pd.Series.to_frame(
                df_song[['title', 'artist']].iloc[random.choice(df_song[df_song['emotion'] == '중립'].index)])
        if pre_ans_str == '거실':
            df_result = pd.Series.to_frame(
                df_song[['title', 'artist']].iloc[random.choice(df_song[df_song['emotion'] == '행복'].index)])
        if pre_ans_str == '공항내부':
            df_result = pd.Series.to_frame(
                df_song[['title', 'artist']].iloc[random.choice(df_song[df_song['emotion'] == '분노'].index)])
        if pre_ans_str == '빵집':
            df_result = pd.Series.to_frame(
                df_song[['title', 'artist']].iloc[random.choice(df_song[df_song['emotion'] == '슬픔'].index)])
        if pre_ans_str == '술집':
            df_result = pd.Series.to_frame(
                df_song[['title', 'artist']].iloc[random.choice(df_song[df_song['emotion'] == '행복'].index)])
        if pre_ans_str == '식당':
            df_result = pd.Series.to_frame(
                df_song[['title', 'artist']].iloc[random.choice(df_song[df_song['emotion'] == '행복'].index)])
        if pre_ans_str == '주방':
            df_result = pd.Series.to_frame(
                df_song[['title', 'artist']].iloc[random.choice(df_song[df_song['emotion'] == '놀람'].index)])
        if pre_ans_str == '컴퓨터실':
            df_result = pd.Series.to_frame(
                df_song[['title', 'artist']].iloc[random.choice(df_song[df_song['emotion'] == '혐오'].index)])
        if pre_ans_str == '창고':
            df_result = pd.Series.to_frame(
                df_song[['title', 'artist']].iloc[random.choice(df_song[df_song['emotion'] == '공포'].index)])
        if pre_ans_str == '침실':
            df_result = pd.Series.to_frame(
                df_song[['title', 'artist']].iloc[random.choice(df_song[df_song['emotion'] == '행복'].index)])

            # print("해당 "+filenames[cnt].split("/")[-1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
        #     print("입력해주신 이미지는 "+pre_ans_str+"로 추정됩니다.")
        # print("==================================================================================")

        cnt += 1
        df_result = df_result.transpose() #웹에 랜더링 형태 편하고 예쁘게 하기위해 행렬전환

    #     print(i.argmax()) #얘가 레이블 [1. 0. 0.] 이런식으로 되어 있는 것을 숫자로 바꿔주는 것.
    # 즉 얘랑, 나중에 카테고리 데이터 불러와서 카테고리랑 비교를 해서 같으면 맞는거고, 아니면 틀린거로 취급하면 된다.
    # 이걸 한 것은 _4.py에.

    #(방법1. 데이터프레임을 table태그로 랜더링하는 방법)
    return render_template('image_model_result.html',
                           column_names=df_result.columns.values,
                           row_data=list(df_result.values.tolist()),
                           zip=zip, pre_ans_str=pre_ans_str)
    # (방법2. 데이터프레임을 to.html()로 랜더링 하는 방법)
    # return render_template('image_model_result.html', tables=[df_result.to_html(classes='data', header="true")])



# ---------------------------------- 서비스 개발중 -----------------------------------------------------
@app.route('/ing')
def ing():
    return "서비스개발중"



# name에 main이 들어가 있는지 확인하는 코드(그냥 실행되기 위한 코드라고 이해)
if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8000, threaded=True, debug=True)  #수정 테스트시 debug모드 ,배포/공유할때는 off할것
    app.run(host='0.0.0.0', port=9000)
