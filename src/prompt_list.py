single_stage_user_prompt_v2_2 = ''' 

Example of incorrect faithful annotation:
###Reference###:
Jacques Michel Gabriel Paul Benoist-Méchin (1 July 1901 – 24 February 1983) was a French far right politician and writer. He was born and died in Paris. Well known as a journalist and historian, he later became prominent for his collaborationism under the Vichy regime. After his conviction in 1947 and release from prison in 1954, he became an Arab world expert in the second part of his life.


###Claims###:
This claim should be tagged as invented: "In 1936, Jacques benoist-méchin joined the french popular party, a far-right political group opposed to democracy." since french popular party is far-right political group opposed to democracy is not mentioned in the reference even though Jacques benoist-méchin was a far right politician.

Example of incorrect subjective annotation:

###Reference###:
Title-safe area 
 
The title-safe area or graphics-safe area is, in television broadcasting, a rectangular area which is far enough in from the four edges, such that text or graphics show neatly: with a margin and without distortion. This is applied against a worst case of on-screen location and display type. Typically corners would require more space from the edges, but due to increased quality of the average display this is no longer the concern it used to be, even on CRTs.

###Claims###:
This claim should be tagged as faithful: Content placed in Title-safe area remains legible and undistorted, ensuring that essential information is accessible to viewers. since the claim can be reasonably inferred from the reference.




You SHOULD refer to the following correct examples:
Example 1:
###Reference###: 
Macedonia competed at the 2016 Summer Paralympics in Rio de Janeiro, Brazil, from 7 to 18 September 2016.

Disability classifications 

Every participant at the Paralympics has their disability grouped into one of five disability categories; amputation, the condition may be congenital or sustained through injury or illness; cerebral palsy; wheelchair athletes, there is often overlap between this and other categories; visual impairment, including blindness; Les autres, any physical disability that does not fall strictly under one of the other categories, for example dwarfism or multiple sclerosis. Each Paralympic sport then has its own classifications, dependent upon the specific physical demands of competition. Events are given a code, made of numbers and letters, describing the type of event and classification of the athletes competing. Some sports, such as athletics, divide athletes by both the category and severity of their disabilities, other sports, for example swimming, group competitors from different categories together, the only separation being based on the severity of the disability.

Shooting

The first opportunity to qualify for shooting at the Rio Games took place at the 2014 IPC Shooting World Championships in Suhl. Shooters earned spots for their NPC.  Macedonia earned a qualifying spot at this event in the P2 – 10m Air Pistol Women SH1  event as a result of Olivera Nakovska-Bikova winning a bronze medal.

See also
Macedonia at the 2016 Summer Olympics

References

Nations at the 2016 Summer Paralympics
2016
2016 in Republic of Macedonia sport
###Claims###:
<>Great questions!</> <>In the paralympics, disability classifications help ensure that competitions are fair and equitable.</> Each sport has its own classification system based on the specific physical demands of the competition. Events are given a code made of numbers and letters to describe the type of event and the classification of the athletes competing.

Some sports, like athletics, divide athletes by both the disability category and the severity of their disabilities. <>This ensures that competitors with similar levels of impairment compete against each other.</> Other sports, like swimming, group athletes from different categories together, with the only separation being based on the severity of the disability. This approach is taken when different disability categories do not significantly impact performance in a given sport.
###Judgement###:
<irrelevant>Great questions!</irrelevant> <subjective>In the paralympics, disability classifications help ensure that competitions are fair and equitable.</subjective> <faithful>Each sport has its own classification system based on the specific physical demands of the competition.</faithful> <faithful>Events are given a code made of numbers and letters to describe the type of event and the classification of the athletes competing.</faithful>

<subjective>Some sports, like athletics, divide athletes by both the disability category and the severity of their disabilities. <subjective>This ensures that competitors with similar levels of impairment compete against each other.</subjective> <faithful>Other sports, like swimming, group athletes from different categories together, with the only separation being based on the severity of the disability.</faithful> <faithful>This approach is taken when different disability categories do not significantly impact performance in a given sport.</faithful>

Example 2:
###Reference###:    
Izumi Station is a railway station on the Jōban Line in the city of Iwaki, Fukushima, Japan, operated by East Japan Railway Company (JR East). The station also has a freight depot for the Fukushima Rinkai Railway Main Line.

Lines
Izumi Station is served by the Jōban Line, and is located 195.0 km from the official starting point of the line at

Station layout
Izumi Station has one island platform and one side platform connected to the station building by a footbridge. The station has a Midori no Madoguchi staffed ticket office.

Platforms

History
Izumi Station opened on 25 February 1897. The station was absorbed into the JR East network upon the privatization of the Japanese National Railways (JNR) on 1 April 1987.

Passenger statistics
In fiscal 2018, the station was used by an average of 2449 passengers daily (boarding passengers only).

Surrounding area
Iwaki-Izumi Post Office
Onahama Port
Aquamarine Fukushima

See also
 List of railway stations in Japan

External links

References

Stations of East Japan Railway Company
Railway stations in Fukushima Prefecture
Jōban Line
Railway stations in Japan opened in 1897
Iwaki, Fukushima
###Claims###:
Of course! <>Izumi station has a combination of one island platform and one side platform, which are connected to the station building by a footbridge.</> <>The island platform allows trains to pass on either side, while the side platform is used for trains arriving from just one direction.</> <>This layout ensures efficient handling of the passenger and freight trains that pass through the station.</>

<>In terms of passenger traffic, izumi station had an average of 2,449 passengers daily (boarding passengers only) during the fiscal year 2018.</> <>This number showcases its significance as a transportation center in the region.</>
###Judgement###:
Of course! <faithful>Izumi station has a combination of one island platform and one side platform, which are connected to the station building by a footbridge.</faithful> <invented>The island platform allows trains to pass on either side, while the side platform is used for trains arriving from just one direction.</invented> <subjective>This layout ensures efficient handling of the passenger and freight trains that pass through the station.</subjective>

<faithful>In terms of passenger traffic, izumi station had an average of 2,449 passengers daily (boarding passengers only) during the fiscal year 2018.</faithful> <subjective>This number showcases its significance as a transportation center in the region.</subjective>

Example 3:
###Reference###:{reference}
###Claims###:{dialogue}
###Judgement###:
'''