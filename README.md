# <b>ProvedArea.py</b>
Travis Salomaki, 2024<br>
ComboCurve, Inc.

## <b>Motivation</b>
The program below presents a Python implementation of SPEE Monograph 3's Expanding Concentric Radii method and introduces a new method of generating a unique probabilistic solution through the use of Monte Carlo sampling.

Since its initial release in 2010, the recommendations introduced by the Society of Petroleum Evaluation Engineer's (SPEE) Monograph 3 have been largely integrated into the key workflows and operating principles of modern exploration and production companies. In chapter three of the monograph, the authors present two methods of determining the statistically proved area of a resource play. For those unfamiliar with the terminology, a <b>resource play</b> can simply be thought of as a regional extent of hydrocarbons exhibiting "low risk and repeatable results," while the term <b>proved</b>, in the context of Oil and Gas reserve evaluations, refers to the "quantity of [hydrocarbons] that a company reasonably expects to extract from a given formation." Specifically, "proven reserves are classified as having a 90% or greater likelihood of being present and economically viable for extraction in current conditions."

The Expanding Concentric Radii Method offers a straightforward means of determining the statistically proved area of a resource play. At a high level, the process involves randomly selecting a group of "anchor" wells within a given population of "analog" wells, generating concentric circles around the anchor wells, and comparing the statistical characteristics of the wells that fall into the concentric circles against those of the analog population until a radial distance is reached in which the associated well characteristics are no longer representative of the analog population. 

At the time of Monograph 3's publication, implementing the Expanding Concentric Radii method in practice would have proven to be largely inconvenient and time consuming due to the limitations of GIS mapping platforms, produciton data access, and manually intensive DCA forecasts. Additionally, given that the anchor wells are randomly selected, the methodology results in a non-unique output that can vary widely from realization to realization. Given the advancements in computational efficiency, the advent of open-source geospatial Python libraries, and the wide-spread adoption of modern auto-forecasting platforms such as ComboCurve, ProvedArea.py provides operators, mineral shops, investment banks, and industry peers alike with a practical means of implementing the Expanding Concentric Radii Method and offers the added benefit of unique probabilistic outputs. 


## <b>Getting Started</b>

For a complete demonstration of the package's functionality, feel free to check out the `ProvedArea.ipynb` file.

#### **Assumptions**
* The well set comprises of a single 'contiguous' drilling area.
* The target resource play is considered to be in the "statistical phase" of maturity. 

#### **Inputs**

All program inputs are taken directly from easily accesible ComboCurve exports. To use ProvedArea.py you'll need the following:
* ComboCurve Well Header Export (.csv)
* ComboCurve Forecast Parameter Export (.csv)

To access the Well Header Export, navigate to the Project Wells tab of your ComboCurve project and click the download icon located in the top right corner of the well header table. To access the Forecast Parameter Export, open a ComboCurve forecast set, click "Forecast Options," and then click "Export Forecast Parameters (CSV)."

Although the program is configured to run using ComboCurve outputs, ProvedArea.py supports outputs from any forecasting platform as long as the file inputs match those expected by the program. This will require the changing of column names and potentially a bit of unit conversion. 

#### **Requirements**

Before you instantiate a ProvedArea object, please make sure that your data set meets the following requirements:
* All wells in the forecast set have a forecast generated for the target phase of interest and have a perforated lateral length (PLL) populated in the well header table. This will allow for the required EUR/PLL values to be populated. 
* All wells need both a surface latitude and a surface longitude field populated in the well header table. 


## <b>Dependencies</b>

ProvedArea.py relies on the following packages:

1. <b>pandas</b>
1. <b>numpy</b>
1. <b>matplotlib</b>
1. <b>geopandas</b>
1. <b>shapely</b>
1. <b>alphashape</b>

If you get a package import error, you may have to first install some of these packages. This can usually be accomplished by opening up a command window on Windows and then typing 'python -m pip install [package-name]'. More assistance is available with the respective package docs.

## References:
1. Guidelines for the Practical Evaluation of Undeveloped Reserves in Resource Plays. SPEE Society of Petroleum Evaluation Engineers, 2010. 
2. https://www.investopedia.com/terms/p/proven-reserves.asp

## Planned Additions
1. More efficient data structure for holding the proved area realizations.
2. Making the package accessible via PyPi. 
