import pandas as pd
import plotly.offline as offline
import plotly.graph_objs as go

class Machine:
    def __init__(self):
        pass
   #     self.code_df = pd.DataFrame({'name':[], 'code':[]})

    def krx_crawl(self):
        self.code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0]
        self.code_df.종목코드 = self.code_df.종목코드.map('{:06d}'.format())
        self.code_df = self.code_df[['회사명','종목코드']]
        self.code_df = self.code_df.rename(columns={'회사명':'name', '종목코드':'code'})
            #tf는 한글인식가능하지만 keras는 한글인식안되므로 영문명으로 변환

    def code_df_head(self):
        print(self.code_df.head())

    def get_url(self, item_name, code_df):
        code = code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False)
        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code='005930') #? 이후의 부분을 query로 치는 것
#        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code) 로 입력해야하는데 삼성전자 코드(005930)를 테스트용으로 집어넣은것
        print('요청 URL = {}'.format(url))
        return url

    # 원래는 변수처리 한 다음에 외부에서 받게하지만 오늘 시스템점검으로 읽어올수 없으므로 '삼성전자'라고 부여
    def test(self, code):
#        item_name = '삼성전자'
 #       url = self.get_url(item_name, self.code_df)
        df = pd.DataFrame()
        for page in range(1, 21):
            pg_url = 'https://finance.naver.com/item/sise_day.nhn?code={code}&page={page}'.format(code=code, page=page)
            df = df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)
            df.dropna(inplace = True) # na : 결측값. null 또는 숫자가 있어야할 자리에 의미없는 값이 있다거나 하는 상태
            return df

    def rename_item_name(self, param): #한글명으로 되어있는 것을 영문명으로 변환
        df = param.rename(columns = {'날짜':'date', '종가':'close', '전일비':'diff',
                                           '시가':'open', '고가':'high', '저가':'low', '거래량':'volumn'})
        df[['close', 'diff', 'open', 'high', 'low', 'volumn']] = \
            df[['close', 'diff', 'open', 'high', 'low', 'volumn']].astype(int)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['date'], ascending=True) # 시계열차트 작성하기 위함. 왼쪽이 가장 오래된 정보
        return df

if __name__ == '__main__':
    m = Machine()
    def print_menu():
        print('0. EXIT\n'
              '1. 종목헤드\n'
              '2. 종목컬럼명 보기\n'
              '3. 전처리완료\n')
        return input('CHOOSE ONE \n')
    while 1:
        menu = print_menu()
        print('MENU %s \n' % menu)
        if menu == '0':
            break
        elif menu == '1':
            m.code_df_head()
        elif menu == '2':
            print(m.test('005930'))
        elif menu == '3':
            print(m.rename_item_name(m.test('005930')))

