from main_model import model_name_RC
import scanpy as sc
import episcanpy as epi

if __name__ == "__main__":
    dataset = 'CITE'  # select sample data type

    if dataset == 'SNARE':
        train_rna_data_path = 'data/SNARE/RNA.h5ad'
        train_atac_data_path = 'data/SNARE/ATAC.h5ad'
        annData_rna = sc.read_h5ad(train_rna_data_path)
        annData_atac = sc.read_h5ad(train_atac_data_path)
        # select 4000 hvgs
        epi.pp.select_var_feature(annData_atac, nb_features=4000, show=False, copy=False)
        sc.pp.highly_variable_genes(annData_rna, n_top_genes=4000, flavor="seurat_v3", subset=True)
        annData_rna.obs['label'] = annData_rna.obs['cell_type']
        annData_atac.obs['label'] = annData_atac.obs['cell_type']
        model_name_RC(annData_rna, annData_atac, dataset)

    if dataset == 'PBMC':
        train_rna_data_path = 'data/PBMC/RNA.h5ad'
        train_atac_data_path = 'data/PBMC/ATAC.h5ad'
        annData_rna = sc.read_h5ad(train_rna_data_path)
        annData_atac = sc.read_h5ad(train_atac_data_path)
        # select 4000 hvgs
        epi.pp.select_var_feature(annData_atac, nb_features=4000, show=False, copy=False)
        sc.pp.highly_variable_genes(annData_rna, n_top_genes=4000, flavor="seurat_v3", subset=True)
        annData_rna.obs['label'] = annData_rna.obs['cell_type']
        annData_atac.obs['label'] = annData_atac.obs['cell_type']
        model_name_RC(annData_rna, annData_atac, dataset)

    if dataset == 'SHARE':
        train_rna_data_path = 'data/SHARE/RNA.h5ad'
        train_atac_data_path = 'data/SHARE/ATAC.h5ad'
        annData_rna = sc.read_h5ad(train_rna_data_path)
        annData_atac = sc.read_h5ad(train_atac_data_path)
        # select 4000 hvgs
        epi.pp.select_var_feature(annData_atac, nb_features=4000, show=False, copy=False)
        sc.pp.highly_variable_genes(annData_rna, n_top_genes=4000, flavor="seurat_v3", subset=True)
        annData_rna.obs['label'] = annData_rna.obs['cell_type']
        annData_atac.obs['label'] = annData_atac.obs['cell_type']
        model_name_RC(annData_rna, annData_atac, dataset)

    if dataset == 'CITE':
        batch = False
        train_rna_data_path = 'data/CITE/RNA.h5ad'
        train_adt_data_path = 'data/CITE/ADT.h5ad'
        annData_rna = sc.read_h5ad(train_rna_data_path)
        annData_adt = sc.read_h5ad(train_adt_data_path)
        doublet_bool = (annData_rna.obs['celltype.l2'] != 'Doublet')
        annData_rna = annData_rna[doublet_bool].copy()
        annData_adt = annData_adt[doublet_bool].copy()
        donors = ['P2']
        index = [x in donors for x in annData_rna.obs['donor']]
        annData_rna = annData_rna[index].copy()
        annData_adt = annData_adt[index].copy()
        # RNA select 4000 hvgs and ADT select all
        sc.pp.highly_variable_genes(annData_rna, n_top_genes=4000, flavor="seurat_v3", subset=True)
        annData_rna.obs['label'] = annData_rna.obs['celltype.l2']
        annData_adt.obs['label'] = annData_adt.obs['celltype.l2']
        model_name_RC(annData_rna, annData_adt, dataset)
