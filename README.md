# Crop Suitability Analysis & Field Validation Brief
**Region**: Iizuka, Sosa City, Chiba Prefecture, Japan  
**Target Crops**: Lemon, Apple, Olive, Soybean  
**Data Source**: 2025 Google Earth Engine (GEE) 10m Grid Export (46,508 points)

---

## 1. Is the System Ready for Field Validation?

**Yes.** The system is fully ready for the next step: **agronomic field validation**. 

### Why it is ready:
1. **High-Resolution Mapping**: By moving the GEE export resolution to **10m**, we are mapping suitability at the individual field/plot scale (necessary for agrivoltaics).
2. **Deterministic Baseline**: The expert-rule suitability scores are computed using agronomic parameters (soil pH, organic matter, temperature, rainfall, and terrain slopes) customized for Japan's high-precipitation environment.
3. **Interactive Visual Tool**: Stakeholders can use the Streamlit app with the new satellite base map to visually locate high-scoring zones, filter by minimum scores, and check details of candidate plots.
4. **Purpose of Field Validation**: The goal of field validation is *not* to verify a perfect model, but to test this "first-screening" baseline. The agronomists/farmers will ground-truth these predictions (e.g., test if a "Priority Zone" actually has the predicted soil pH and is growing healthy crops), providing the critical labels needed to calibrate the model.

---

## 2. Sosa (Iizuka) Case Study Results

The analysis of **46,508 grid points** across the Iizuka, Sosa bounding box (`35.718° to 35.746° N`, `140.535° to 140.569° E`) reveals the following environmental and crop-specific patterns:

### Local Environmental Summary (Mean Values)
*   **Annual Rainfall**: `1,492 mm` (High moisture, typical for coastal Chiba).
*   **Mean Temp**: `16.86°C` (Warm temperate climate).
*   **Soil pH**: `5.80` (Slightly acidic, typical for volcanic ash/Andosols).
*   **Soil Organic Carbon**: `8.69` (Moderate-low mineral soil profile).
*   **Elevation**: `20.27 m` (Low coastal plain).
*   **Slope**: `6.54°` (Gentle rolling terrain).

### Crop Suitability Statistics
The table below compares the rule-based suitability scores and the machine learning model approximations:

| Crop | Mean Suitability (0-100) | High Suitability (Score $\ge 65$) | Priority Trial Zones (Score $\ge 80$) | Best Crop Count (Locations Won) |
| :--- | :---: | :---: | :---: | :---: |
| **🫒 Olive** | **68.24** | **29,058 points (62.5%)** | **16,206 points (34.8%)** | **27,279 points (58.7%)** |
| **🍋 Lemon** | **65.47** | **25,534 points (54.9%)** | **7,265 points (15.6%)** | **13,907 points (29.9%)** |
| **🌱 Soybean** | **56.17** | **8,246 points (17.7%)** | **701 points (1.5%)** | **5,322 points (11.4%)** |
| **🍎 Apple** | **52.58** | **9,431 points (20.3%)** | **0 points (0.0%)** | **0 points (0.0%)** |

### Agronomic Analysis of the Results

1. **Olive (Winner - 58.7% of locations)**:
   - *Why?* Olive is highly compatible with the gentle slopes ($6.54°$) which allow for good drainage, preventing waterlogging despite the high rainfall ($1,492$ mm). The temperature profile is optimal, and olives tolerate the slightly acidic soil pH of $5.8$.
2. **Lemon (High Suitability - 29.9% of locations)**:
   - *Why?* Citrus crops grow well in Kanto's warm coastal temperatures. The soil pH of $5.8$ falls right into the lemon's preferred range ($5.5$–$6.5$). The high rainfall provides plenty of water, though winter frost risk (unmodeled) must still be assessed on-site.
3. **Soybean (Moderate Suitability - 11.4% of locations)**:
   - *Why?* Soybeans are widely grown under solar sharing structures in Sosa. However, the global soil pH proxy ($5.80$) is slightly below the soybean's optimal nodulation range ($6.2$–$7.2$). Soybeans also prefer completely flat fields; slopes above $8°$ lower their score. *Note: Actual fields likely have higher pH due to agricultural liming, which will be corrected during field validation.*
4. **Apple (Low Suitability - 0.0% of locations won)**:
   - *Why?* Apples require significant winter chilling and prefer cooler climates ($8$–$15°$C mean annual temp). Sosa's mean temp of $16.86°$C is too warm for commercial apple production, restricting the crop to a moderate-low score with no priority zones.

---

## 3. Study Period & Data Limitations

When presenting these results to stakeholders, it is important to communicate the following limitations:

*   **Study Period**: The analysis uses satellite composites and climatic data from the year **2025**. Inter-annual climate variability (e.g., an unusually dry or cold year) is not captured.
*   **Macro-Scale Proxies**: Soil pH ($5.8$) and Organic Carbon ($8.7$) are derived from global OpenLandMap datasets. They do not capture micro-scale farmer interventions, such as liming (which raises pH) or compost addition.
*   **Macro-Climate Data**: ERA5 temperature and CHIRPS precipitation have a spatial resolution of several kilometers. Local micro-climates (wind breaks, frost pockets, coastal fog) are not represented.
*   **Agrivoltaic Shading**: The suitability score evaluates open-field conditions. It does not yet account for the reduction in solar radiation (shading) or change in soil moisture directly under solar panels.
*   **Pseudo-Supervised Nature**: The Random Forest model is trained to reproduce the expert rules. It is a proxy of our assumptions, not a predictor of real-world crop yield.

---

## 4. Future Enhancements

We are planning the following enhancements to mature the PoC into a production-grade system:

1. **Ground-Truth Calibration (Liming Correction)**:
   - Collect soil samples from the trial plots in Sosa.
   - Replace the OpenLandMap soil layers with local soil sample interpolations to correct pH and organic matter values.
2. **Agrivoltaic Shading & Micro-Climate Model**:
   - Add a shading coefficient layer based on panel density (e.g., $30\%$ shading).
   - Adjust the crop profiles to evaluate suitability under partial shade (e.g., soybeans have high shade tolerance, while lemons have low shade tolerance).
3. **On-Site Weather Station Integration**:
   - Connect local automated weather station feeds to replace regional ERA5 temperature data, capturing exact frost dates.
4. **Farmer Feedback Interface**:
   - Build a web portal where farmers can input actual crop health observations, yield data, and photos, allowing the Random Forest model to transition from pseudo-supervised to true supervised learning.
