python3 train.py \
    --train_data "TBD" \
    --valid_data "TBD" \
    --select_data / \
    --batch_ratio 1 \
    --Transformation "TPS" \
    --FeatureExtraction "ResNet" \
    --SequenceModeling "BiLSTM" \
    --Prediction "Attn" \
    --num_iter 50000 \
    --valInterval 200 \
    --data_filtering_off \
    --FT \
    --character " 0123456789가강거경계고관광구금기김나남너노누다대더도동두등라러로루마머명모무문미바배버보부북사산서소수아악안양어연영오용우울원육인자작저전조주중지차천초추충카타파평포하허호홀제"

