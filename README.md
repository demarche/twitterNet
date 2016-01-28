# twitterNet
卒検
RT数予測

## 準備
### Step1. ツイート収集
Tweet crawler with image  
<https://github.com/demarche/TweetCrawlerWithImage>  
収集にはtwitterプロフィールのjsonが必要    

### Step2. Doc2Vecモデルを用意
あとで

### Step2. 画像付きツイートを学習用形式に変更
`python Convolution2d.py -k 2 -p <user_info.txtが保存されている場所>`  
画像やRT数が保存される

## 学習
`python Convolution2d.py -k 0 -p <学習用データの保存場所> -s <学習済みデータ> -g <gpu id> -r <回帰問題>`  
学習済みデータ…ない場合省略可  
gpu id…-1でcpu処理  
回帰問題…回帰問題で学習する場合は1  

## テスト
`python Convolution2d.py -k 1 -p <学習用データの保存場所> -s <学習済みデータ> -g <gpu id> -r <回帰問題>`  
