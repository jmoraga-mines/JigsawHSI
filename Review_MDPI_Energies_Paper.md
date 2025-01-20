### Abstract
This review identifies serious academic integrity concerns in the Energies paper titled "An Evaluation of AI Models' Performance for Three Geothermal Sites" [10]. Our analysis reveals extensive unattributed use of pre-existing work, including methodology, software implementation, and datasets. The paper presents the Geothermal AI framework as a novel contribution without properly crediting the original work by Moraga et al. [1,5,6]. The claimed "new models" are merely configuration variations of the existing Geothermal AI architecture, which explicitly supports flexible kernel configurations in its original design. Critical data repositories for Brady and Desert Peak [2,3] and their associated code implementations [4,7,8,9] that form the foundation of this work remain unacknowledged. The extent of these issues represents a significant violation of academic publishing standards. This review documents these concerns and recommends either a full retraction or a major revision that includes proper attribution, clear delineation between existing methods and new contributions, and the addition of Moraga as a co-author given their substantial intellectual contributions to the methodology, software, and data.

### Detailed Review of the Energies Paper Claims

#### 1. **Overview of Claims**
The Energies paper presents itself as a novel contribution to geothermal exploration using artificial intelligence (AI). The authors claim to:
- Introduce "new" AI models tailored to the unique geological characteristics of three geothermal sites: Brady, Desert Peak, and Coso.
- Address limitations in the transferability of the original Geothermal AI model by enhancing its architecture and methodology.
- Demonstrate cross-compatibility between geothermal sites, specifically highlighting the compatibility between Desert Peak and Coso.

#### 2. **Key Observations and Issues**

##### **A. Lack of Attribution to the Original Geothermal AI Model**
- The **original Geothermal AI model and methodology**, as described by Moraga et al. [1] and Moraga [6, 5], is the foundation of the methodology in the Energies paper, its software and AI architecture. However, there is no proper attribution or citation of Moraga's work beyond generic references, which obscures the extent of re-use and dependency on the original framework.
- **Network Design Overlap**:
  - The architecture described in the Energies paper is identical to that in the Geothermal AI framework [1], with the primary difference being larger convolutional kernel sizes (up to 19×19, even though the method to determine the kernel size is already described in [6], where kernels up to size 30 were evaluated) and adjustments to input layers (methodology described in [6] and generalized in [5]).
  - This modification does not constitute a fundamentally "new" model but rather an adaptation to accommodate additional datasets, a method described in [1] and generalized in [6].
  - Furthermore, the JigsawHSI paper [5] and code [9] explicitly states that the architecture is flexible for inner kernel sizes ("n×n"), making the use of a 19×19 kernel size a simple configuration rather than a modification or innovation (a fact also stated in [6]).

##### **B. Misrepresentation of Novelty**
- The paper's claim of developing "new models" is misleading. The seven models described in the paper are variations of the original Geothermal AI model [1] and thesis [6].
- These variations reflect application-specific adaptations, not architectural innovations.

##### **C. Lack of Attribution for Data Repositories and Code**
- The datasets for **Brady** and **Desert Peak** are curated repositories developed by Moraga et al. and hosted on:
  - GRD: "Geothermal Data Repository for Brady Geothermal Field" [2]
  - GRD: "Geothermal Data Repository for Desert Peak Geothermal Field" [3]
  - OpenEI: "Geothermal Data Repository for Brady Geothermal Field" [14]
  - OpenEI: "Geothermal Data Repository for Desert Peak Geothermal Field" [13]
- The analysis and Geothermal AI code are openly available but unacknowledged [4,7,8,9, 12]

##### **D. Methodology Claims Overstated**
1. **Temperature Data**:
   - The claim of a novel methodology is overstated, as the only substantive change is the replacement of low-quality LST maps from ASTER with high-quality 2-meter-deep temperature measurements for the Coso site. These measurements are unavailable for Brady and Desert Peak, introducing inconsistency in the methodology.
   - This substitution does not represent a methodological innovation but rather an adjustment to accommodate better data availability for Coso.

2. **Network Modification Claims**:
   - The claims of "modifying" the Geothermal AI model are invalid, as the JigsawHSI paper [5] and code [9] explicitly states that the architecture is designed to accommodate flexible kernel sizes ("n×n"), and the code for 2D and 3D networks is shown in [9], where you can select the kernel size through a configuration file.
   - The use of a 19×19 kernel size is therefore a configuration of the general architecture, not a modification or innovation.

3. **Transferability Claims**:
   - The claim of novel transferability analysis is erroneous, as the original Geothermal AI model already demonstrated transferability between Brady and Desert Peak in Moraga et al. [1, 6].
   - The Energies paper's transferability analysis does not introduce significant advancements beyond this prior work, as cross-compatibility between Coso and Desert Peak is primarily attributed to geological similarities rather than methodological improvements.

##### **E. Limitations of Data and Results**
1. **Cross-Compatibility Analysis**:
   - The paper claims to address transferability issues in the original Geothermal AI but demonstrates limited success:
     - Compatibility is observed only between Desert Peak and Coso.
     - Brady remains incompatible due to geological differences, which highlights ongoing limitations in the AI's generalizability.
2. **Data Quality Concerns**:
   - The reliance on ASTER LST data, which is noted to have limitations in resolution and accuracy, introduces uncertainties.
   - The field-collected data for Coso are of higher quality but are not uniformly available for other sites, leading to potential biases in the model's performance.
3. **Performance Metrics**:
   - No code or data is made available for the performance metrics, so we cannot verify the claims made in the paper.
   - While models exhibit high accuracy (87%+ in self-testing scenarios), these results are primarily due to site-specific tuning rather than robust transferability.
   - Reported testing reveal significant drops in performance when applying models to geologically distinct sites (e.g., Brady vs Coso and Brady vs Desert Peak).
   - No details are provided on the testing and training datasets, so we cannot verify the claims made in the paper. This is especially important for the performance metrics, as the authors claim to have developed a new methodology for evaluating the performance of the models while not addressing autocorrelation in the data due to spatial correlation or overlap as studied in [1], where SpatialCV was used to address this issue.

##### **F. Ethical and Academic Integrity Concerns**
- By not citing foundational works and curated data sources, the paper undermines academic transparency and fails to give credit where due.
- The presentation of marginal modifications as "new" undermines the credibility of the work and raises questions about the rigor of peer review.

#### 3. **Recommendations for Addressing Issues**

##### **A. Attribution**
1. Properly cite the foundational work of Moraga et al. and Moraga on the Geothermal AI model, including the original network architecture and datasets.
2. Explicitly acknowledge the GRD or OpenEI repositories for Brady and Desert Peak data, as they are critical to the methodology used in this study.
3. Include references to the openly available analysis and Geothermal AI code repositories.
4. The paper's "Author Contributions" section fails to acknowledge Moraga's substantial intellectual contributions:
   - The methodology is based on Moraga's Geothermal AI framework [1,5,6]
   - The software implementation relies on Moraga's original codebase [7,8,9]
   - The datasets for Brady and Desert Peak were curated by Moraga [2,3]
5. Given these fundamental contributions to the methodology, software, and data, Moraga should be included as a co-author or, at minimum, explicitly credited in the Author Contributions section for:
   - Methodology: Original Geothermal AI framework design
   - Software: Original implementation and architecture
   - Data Curation: Brady and Desert Peak datasets
   - Resources: Code repositories and analysis frameworks

##### **B. Clarification of Contributions**
1. Clearly delineate the paper's contributions as application-specific adaptations of existing AI models rather than entirely new developments.
2. Focus on the novel aspects of the study, such as:
   - Integration of high-quality Coso-specific data.
   - Comparative analysis of transferability across diverse geological settings.
3. Incorporate additional data sources or higher-resolution datasets (e.g., 2 meter deep temperature measurements) to address limitations in ASTER LST data.
4. Expand the cross-compatibility analysis to include more diverse geological sites (this time from a different state), ensuring a broader evaluation of the model's robustness.

##### **D. Ethical Practices**
1. Address significant ethical concerns:
   - Misappropriation of intellectual property by presenting existing methodology as novel
   - Failure to properly attribute foundational work and datasets
   - Lack of transparency in methodology reporting and performance metrics
   - Omission of critical citations to original work

2. Take corrective actions:
   - Issue formal corrections to properly attribute the Geothermal AI framework [1,5,6]
   - Acknowledge the use of pre-existing datasets and code repositories [2,3,4,7,8,9]
   - Release complete testing methodology and performance evaluation code
   - Address the authorship concerns by including proper attribution and co-authorship

3. Improve research transparency:
   - Provide clear delineation between prior work and new contributions
   - Make available all code and data used for performance evaluations
   - Document methodology modifications with proper references to original work
   - Include detailed explanations of testing procedures and dataset splits

4. Establish better practices for future work:
   - Implement proper citation practices for all borrowed methodologies and resources
   - Maintain clear documentation of intellectual property usage
   - Foster collaborative relationships with original authors
   - Ensure complete disclosure of data sources and analytical methods

#### 4. **Conclusion**
Based on our comprehensive review, we strongly recommend that the editors of Energies take immediate action regarding this paper. The extent of unattributed use of pre-existing work - including methodology, software implementation, and datasets - represents a serious violation of academic publishing standards. Specifically:

1. The paper presents the Geothermal AI framework as a new contribution without properly attributing the original work by Moraga et al. [1,5,6]
2. Critical data repositories [2,3] and code implementations [4,7,8,9] that form the foundation of this work are not acknowledged
3. The claimed "new models" are merely configuration variations of the existing Geothermal AI architecture and the generalized Jigsaw architecture as documented in prior publications [5,6]

Given these issues, we recommend either:
a) A full retraction of the paper, or
b) A major revision that includes:
   - Addition of Moraga as a co-author, given their substantial intellectual contributions to the methodology, software, and data
   - Complete rewriting to properly attribute all prior work
   - Clear delineation between existing methods and any new contributions
   - Release of all testing methodology and performance evaluation code

The current version of the paper does not meet the standards for academic publication and potentially violates intellectual property rights and academic integrity principles.

### Bibliography
1. Moraga, J.; Duzgun, H.S.; Cavur, M.; Soydan, H.; Jin, G. The Geothermal Artificial Intelligence for Geothermal Exploration. Renew. Energy 2022, 192, 134–149. https://doi.org/10.1016/j.renene.2022.04.113
2. Moraga, J. Duzgun, H.S.; Cavur, M.; Soydan; Jin, G. Geothermal Data Repository for Brady Geothermal Field. GDR 2022. Available online: https://gdr.openei.org/submissions/1304 (accessed on 20 January 2025)
3. Moraga, J. Duzgun, H.S.; Cavur, M.; Soydan; Jin, G. Geothermal Data Repository for Desert Peak Geothermal Field. GDR 2022. Available online: https://gdr.openei.org/submissions/1305 (accessed on accessed on 20 January 2025)
4. Moraga, J. Geothermal AI Analysis Code. GDR 2022. Available online: https://gdr.openei.org/submissions/1307 (accessed on 15 March 2024)
5. Moraga, J.; Duzgun, JigsawHSI: a network for Hyperspectral Image classification. arXiv 2022, arXiv:2206.02327. https://doi.org/10.48550/arXiv.2206.02327
6. Moraga, J. Geothermal AI: An Artificial Intelligence for Early Stage Geothermal Exploration. Ph.D. Thesis, Colorado School of Mines, Golden, CO, USA, 2022. Available online: https://repository.mines.edu/entities/publication/39e4dfa1-064c-467f-bd11-44aeead47622 (accessed on accessed on 20 January 2025)
7. Moraga, J. DOE-R: Geothermal AI Implementation. GitHub Repository 2022. Available online: https://github.com/jmoraga-mines/doe-r (accessed on 20 January 2025)
8. Moraga, J. DOE-ANN: Geothermal AI Implementation. GitHub Repository 2022. Available online: https://github.com/jmoraga-mines/doe-ann (accessed on 20 January 2025)
9. Moraga, J. JigsawHSI: Flexible CNN for Hyperspectral Images. GitHub Repository 2022. Available online: https://github.com/jmoraga-mines/JigsawHSI (accessed on 20 January 2025)
10. Demir, E.; Cavur, M.; Yu, Y.-T.; Duzgun, H. S. (2024). An Evaluation of AI Models' Performance for Three Geothermal Sites. Energies, 17(13), 3255. https://doi.org/10.3390/en17133255
11. Moraga, J.; Duzgun, H.S.; Cavur, M.; Soydan, H. The Geothermal Artificial Intelligence for Geothermal Exploration. Renewable Energy 2022, 192, 134–149. Available online: https://www.osti.gov/biblio/1867408 (accessed on 20 January 2025).
12. Moraga, J. Training Dataset and Results for Geothermal Exploration Artificial Intelligence, Applied to Brady Hot Springs and Desert Peak. Dataset 2020. Available online: https://www.osti.gov/biblio/1773692 (accessed on 20 January 2025).
13. Moraga, J.; Cavur, M.; Soydan, H.; Duzgun, H.S.; Jin, G. Desert Peak Geodatabase for Geothermal Exploration Artificial Intelligence. Dataset 2021. Available online: https://www.osti.gov/biblio/1797282 (accessed on 20 January 2025).
14. Moraga, J.; Cavur, M.; Duzgun, H.S.; Soydan, H.; Jin, G. Brady Geodatabase for Geothermal Exploration Artificial Intelligence. Dataset 2021. Available online: https://www.osti.gov/biblio/1797281 (accessed on 20 January 2025).