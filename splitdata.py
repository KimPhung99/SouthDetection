import os
from scipy.io import wavfile

root = '.'

duration = 4  # so giay
labels = ['normal', 'deep', 'strong', 'none']
source_dir = '{}/datawav_filter'.format(root)
train_dir = '{}/data/training'.format(root)
test_dir = '{}/data/testing'.format(root)

# kiem tra duong link da co hay chua
assert not os.path.isdir(train_dir), "Error! train_dir already exist - {}".format(train_dir)
assert not os.path.isdir(test_dir), "Error! test_dir already exist - {}".format(test_dir)

train_spk = [
    "01_male_23_BQuyen",
    "02_male_22_PTuan",
    "03_male_21_BDuong",
    "04_female_21_LAnh",
    "05_male_21_NLinh",
    "06_male_21_QViet",

    "07_male_21_MQuang",
    "08_male_21_TLong",
    "09_male_21_Ngon",
    "10_male_21_Nam",
    "11_female_21_Tam",
    "12_male_21_Tam",
    "13_female_20_TNhi",
    "14_male_21_Khanh",

    "15_female_21_PPhuong",
    "16_male_21_TTung",
    "17_male_21_Trung",
    "18_male_21_Hoa",
    "19_male_21_Minh_no-ok",
    "20_male_21_Viet",
    "21_male_21_Hai",
    "22_male_21_VHung",
    "23_male_21_CNDuong",
    "24_female_21_MPham",
    "25_famale_21_TCuc_sickness",
    "26_female_19_Linh",

]

test_spk = [
    "27_female_19_TThanh",
    "28_male_19_VHoa_asthma",
    "29_male_19_Cong",
]

#tao duong dan lay label
for label in labels:
    indir = os.path.join(source_dir, label)
    outdir_train = os.path.join(train_dir, label)
    outdir_test = os.path.join(test_dir, label)
    assert os.path.exists(indir), "Error! {} not found".format(indir)
    os.makedirs(outdir_train) #tao folder
    os.makedirs(outdir_test)
    for f in os.listdir(indir):
        if ".wav" in f:
            infile = os.path.join(indir, f) #tao dg dan den file
            #tach data 4s 1 vs data la so die du lieu

            rate, data = wavfile.read(infile)
            if len(data) == duration * rate:
                spk = "_".join(f.replace(".wav", "").split("_")[:-2])
                if spk in train_spk:
                    outfile = os.path.join(outdir_train, f) # tao ten link
                    os.symlink(os.path.realpath(infile), outfile) # tao sort cut
                    print(os.path.realpath(infile), outfile)
                elif spk in test_spk:
                    outfile = os.path.join(outdir_test, f)

                    os.symlink(os.path.realpath(infile), outfile)
                    print(os.path.realpath(infile), outfile)