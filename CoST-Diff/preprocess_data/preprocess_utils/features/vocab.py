import os, json

def build_uli_index(df_bs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    uli_index = df_bs['c_uli'].astype(str).reset_index(drop=True)
    uli_index_dict = {uli: i for i, uli in enumerate(uli_index)}

    with open(os.path.join(out_dir,'uli_index_dict.json'),'w') as f:
        json.dump(uli_index_dict, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir,'index_uli_dict.json'),'w') as f:
        json.dump({i: uli for uli,i in uli_index_dict.items()}, f, indent=2)

    return uli_index_dict
