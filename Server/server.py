from flask import Flask,request,jsonify
app=Flask(__name__)
import json
import sarcasm_predictor
import toxic_predictor
@app.route('/hello')
def hello():
    return "Hi"

@app.route('/sarcasm_predict',methods=['POST'])
def sarcasm_predict():
    sentance=request.json['string']
    print(sentance)
    try:
       ans=sarcasm_predictor.predict_sarcasm(sentance)
       response=jsonify({
        'is_sarcastic':ans
        })
    except:
        return "Could not resolve"
    return response


@app.route('/toxicity_predict',methods=['POST'])
def toxicity_predict():
    sentance=request.json['string']
    print(sentance)
    ans=toxic_predictor.predict_toxicity(sentance)
    try:
       response=jsonify({'toxic':ans})
       response.headers['Access-Control-Allow-Origin'] = '*'
       response.headers['Access-Control-Allow-Credentials'] = 'true'
       response.headers['Access-Control-Allow-Headers'] = 'Origin,Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,locale'
       response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
       
    except:
        return "Could not resolve"
    return response





if __name__=="__main__":
    print("Server Running")
    app.run(host='192.168.127.20',debug=True, port=3000)