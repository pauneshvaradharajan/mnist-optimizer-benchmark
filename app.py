from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
st.set_page_config(page_title='CNN Optimizer Benchmark',page_icon='📊',layout='wide')
st.title('CNN Optimizer Benchmark')
st.caption('A first-semester study comparing optimizer behavior on a compact CNN for MNIST classification.')

results=pd.read_csv(RESULTS/'original_notebook_results.csv')
losses=pd.read_csv(RESULTS/'original_notebook_epoch_losses.csv')
with st.sidebar:
    st.header('Explore the recorded run')
    sort_by=st.selectbox('Rank optimizers by', ['Test accuracy','Training time'])
    include_lsgd=st.toggle('Include custom L-SGD',value=True)
    st.markdown('**Source:** saved outputs from the original semester notebook. These are reported observations, not reproduced results.')

shown=results.copy()
if not include_lsgd: shown=shown[shown.optimizer!='L-SGD']
if sort_by=='Test accuracy': shown=shown.sort_values('accuracy_pct',ascending=False)
else: shown=shown.sort_values('training_time_s',ascending=True)

best=results.loc[results.accuracy_pct.idxmax()]
lsgd=results.loc[results.optimizer=='L-SGD'].iloc[0]
col1,col2,col3=st.columns(3)
col1.metric('Highest recorded accuracy',f'{best.accuracy_pct:.2f}%',best.optimizer)
col2.metric('L-SGD recorded accuracy',f'{lsgd.accuracy_pct:.2f}%')
col3.metric('Training subset',f'{int(results['train_samples'].iloc[0]):,} images','20% of MNIST train split')

st.subheader('Recorded optimizer comparison')
chart_metric=st.radio('Chart metric',['Test accuracy (%)','Training time (seconds)'],horizontal=True)
chart=shown.set_index('optimizer')[['accuracy_pct' if chart_metric.startswith('Test') else 'training_time_s']]
chart.columns=[chart_metric]
st.bar_chart(chart,color='#285F8F')
st.dataframe(shown[['optimizer','accuracy_pct','training_time_s','rss_change_mb']].rename(columns={'optimizer':'Optimizer','accuracy_pct':'Test accuracy (%)','training_time_s':'Training time (s)','rss_change_mb':'Process RSS change (MB; two runs only)'}),hide_index=True,width="stretch")

with st.expander('Training loss by epoch'):
    loss_view=losses[losses.optimizer.isin(shown.optimizer)]
    st.line_chart(loss_view.pivot(index='epoch',columns='optimizer',values='train_loss'),y_label='Mean training loss')

st.subheader('What was compared')
st.markdown('The original notebook uses a two-convolution CNN, a random 20% subset of the MNIST training data, a batch size of 64, five epochs, and the full 10,000-image MNIST test split. It records test accuracy and elapsed training time for SGD, momentum SGD, Nesterov SGD, Adam, RMSprop, Adagrad, Adadelta, AdamW, and a custom L-SGD optimizer.')
st.info('Interpret these figures as one historical notebook run. The original run did not fix random seeds or repeat experiments. The reported RAM values are process-level RSS changes from only the SGD and L-SGD runs; they are not a direct measurement of optimizer-state memory or embedded-device RAM savings.')
st.warning('The supplied course report describes Fashion-MNIST, seven epochs, Keras/PyTorch, and STM32 deployment. The supplied notebook instead contains a PyTorch MNIST experiment with five epochs and no firmware/deployment code. This demo follows the notebook evidence and makes no STM32 deployment claim.')

st.subheader('Run a reproducible experiment')
st.code('pip install -r requirements-training.txt\npython src/train.py --epochs 5 --subset-fraction 0.2 --seed 42',language='bash')
st.download_button('Download recorded notebook metrics',results.to_csv(index=False).encode('utf-8'),'original_notebook_results.csv','text/csv')
st.markdown('Source notebook is archived in the GitHub repository. See the README for methodology, caveats, and reproduction steps.')

