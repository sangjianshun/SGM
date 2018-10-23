# SGM
SGM全称sequence generation model是一种端到端的多标签分类模型。本文介绍SGM的实现细节。其中SGM使用pytorch框架实现。
# 一、训练预料预处理
## 对文本进行预处理就是常规的处理方式：对文章里面的每一个最小单元（如果是分词就是词语）进行one-hot编码，即以一个数字代替文章里面出现的最小单元（对文章进行向量转化）。

## 对标签进行预处理，和文本的初始化一样，对标签的每一个情况进行one-hot编码，除开语料库中的标签以外，原文中还引入了四个其他标签：如2表示文章其实标签，3表示文章的结束标签。SGM就是根据3来实现不同标签长度的输出的，即舍弃3之后的所有预测。

# 二、模型架构
SGM模型是一种seq2seq模型，引入了attention机制，并且考虑了class与class之间的关系。

## 1、SGM模型的输入
```
def forward(self, src, src_len, tgt, tgt_len):
```
src表示经过padding的文章（max_len，batch_size）,其中padding的过程在载入数据的过程中执行。src_len表示每一篇文章的长度（拥有词语的数目）。tgt表示文章的经过padding之后的标签（max_len+2,batch_size）。加2的原因是因为在预处理过程中，对于每一篇文本的标签前面加上了一个2作为起始，末尾加上3作为结束。由于SGM可以表示不同长度的标签，所以引入tgt_len，这和src_len的意义一致。

## 2、过程
### 2.1、预处理
首先按照每一篇文章的长度对batch_size篇文章进行排序，然后按照索引返回新的src和tgt

```
lengths, indices = torch.sort(src_len.squeeze(0), dim=0descending=True)
src = torch.index_select(src, dim=1, index=indices)
tgt = torch.index_select(tgt, dim=1, index=indices)
```


### 2.2、encoder层
#### 2.2.1、输入
SGM的encoder层使用了双向LSTM，其中encoder层的输入为

```
def forward(self, input, lengths):
```
input和整个模型的输入src一致，lengths是src_len转化为list类型的结果。
#### 2.2.2、过程


==步骤一：将输入的batch_size篇文章向量化==

```
self.embedding = nn.Embedding(vocab_size, config.emb_size)#随机初始化
input = self.embedding(input)
```
pytorch中拥有pack_padded_sequence 和pad_packed_sequence函数可以很高效的处理padding对LSTM造成影响。所以产生辅助过程：

```
embs = pack_padded_sequence(input, lengths)
```
++注意此时的embs的数据项，embs.data是一个二维数据，（，emb_size）第一个维度的大小是batch_size篇语料库中的所有词语个数，即padding之前的效果++

==步骤二：将数据进行双向LSTM计算，每一个词语产生一个正向和反向输出`$h_i$`==

```
self.rnn = nn.LSTM(input_size=256，hidden_size=config.encoder_hidden_size=256,
                   num_layers=config.num_layers=2, dropout=config.dropout=0.5，bidirectional=config.bidirec)
outputs, （h,c） = self.rnn(embs)


```
此时输出outputs.data的第一个维度和embs的第一个维度一样，第二个维度是隐藏层的维度的2倍（由于是双向LSTM，所以此处直接将两个`$h_i$`合并在了一起，变成了隐藏层维度的两倍），state是一个tuple类型，（h,c）=state。h有三个维度，第一个维度是LSTM层数（每一层都有一个h的输出），第二个维度是batch_size的大小（一篇文章最后会产生），第三个维度是LSTM隐藏层的维度的2倍，因为是双向LSTM。

```
outputs = unpack(outputs)[0]
batch_size = h.size(1)
h = h.transpose(0, 1).contiguous().view(batch_size, -1, 2 * config.encoder_hidden_size)
c = c.transpose(0, 1).contiguous().view(batch_size, -1, 2 * config.encoder_hidden_size)
state = (h.transpose(0, 1), c.transpose(0, 1))
```
#### 2.2.3、输出

```
return outputs, state
```
outputs.data一共有三个维度，前面两个维度和src的维度一致（src为embedding之前所以只有两个维度），第三个维度为LSTM最后一层隐藏层的输出（论文应该是双向，应该是两倍的这个维度）。

state =（h，c）h和c有三个维度，第一个维度是2(双向LSTM)，第二个维度是batch_size,第三个维度是2倍的LSTM最后一层输出维度。

### 2.3、decoder层
#### 2.3.1、输入

```
def forward(self, inputs, init_state, contexts):
```
inputs表示标签，由于每篇文章的标签在首部和尾部增加了2和3所以inputs需要在tgt的基础上去掉最后一个标签，init_state就是encoder层的state，context是encoder层的outputs交换了第一个维度和第二个维度，故context还是有三个维度，第一个维度是batch_size，第二个维度是padding之后的每篇文章的长度，第三个维度LSTM计算后的输出维度（每一个词语的特征维度）。
#### 2.3.2、过程
==步骤三：和src一样，需要对标签inputs进行维度映射==

```
embs = nn.Embedding(tgt_vocab.size(), config.emb_size)(tgt[:-1])
```
embs的维度为3，第一个维度是标签的个数+1（首部添加了起始符2），第二个维度是batch_size，第三个维度是映射的维度。

==步骤四：以起始符号开始，以标签作为输入使用LSTM==

```
for emb in embs.split(1):
    print(1)
    break
```
emb表示起始符2,维度为三维，第一个维度为1所以需要squeeze（0），第二个维度为batch_size，第三个维度为映射之后的维度

```python
self_layers=[]
for i in range(2):
    #config.emb_size=256.config.decoder_hidden_size=512
    self_layers.append(nn.LSTMCell(input_size=config.emb_size, hidden_size=config.decoder_hidden_size))
    input_size = config.decoder_hidden_size

h_0, c_0 = state
h_1, c_1 = [], []
for i, layer in enumerate(self_layers):
    h_1_i, c_1_i = layer(emb.squeeze(0), (h_0[i], c_0[i]))
    input = h_1_i
    if i + 1 != 2:
        input = nn.Dropout(0.5)(input)
    h_1 += [h_1_i]
    c_1 += [c_1_i]

h_1 = torch.stack(h_1)
c_1 = torch.stack(c_1)
return input, (h_1, c_1)
```
其中此处建立了两个LSTM单元，第一个LSTM单元的维度是（256,512）tgt被映射成了256维，第二个LSTM单元的维度是（512,512）（之所以要两层LSTM，是因为要考虑到前向后向的h和c）。返回的output是第二层的h，state是两层的h和c的stack之后的结果，方便下次使用，和多个输入的LSTM有一点相似，这里的输入只有一个，但是h和c有两个。

==步骤五：计算attention机制里面的权值，根据步骤四的LSTM得出的input（LSTM的h输出）和在encoder层得出的每一个词语的特征向量（双向LSTM的h的stack）==

```
self_linear_in = nn.Linear(512, 512)

gamma_h = self_linear_in(input).unsqueeze(2)
weights = torch.bmm(contexts, gamma_h).squeeze(2)
weights = nn.Softmax(weights)
```
其中input是步骤四的输出结果，维度为(batch_size,N),N为LSTM的输出维度512，contexts是encoder的输出outputs，拥有三个维度，第一个维度是batch_size，第二个维度是src经过padding之后的长度，第三个维度是N=512。

==步骤六：根据权值计算出`$c_t$`==

```
c_t = torch.bmm(weights.unsqueeze(1), contexts).squeeze(1)
```
==步骤七：根据`$c_t$`和`$y_{told}$`(`$s_t$`)计算`$y_t$`在高维上的特征表示==

```

output = nn.Tanh()(nn.Linear(2*512, 512)(torch.cat([c_t, input], 1)))
```
==步骤八：最后一步很简单，只需要对步骤七的结果映射到指定类别个数的维度（如最后每一类可以选80类，则最后全连接到80即可），然后使用softmax选择预测==

#### 2.3.3、输出
encoder层的输出是标签在高维空间的表示。未使用全连接层进行分类。
