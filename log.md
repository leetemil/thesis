## 2019-05-09: Møde med Wouter.
- Probabilistic Programming: Der er en nice guy der hedder Frank Wood der laver nogle vilde ting. Ham kan vi kigge på hvis vi gider.
- Buzzwords: Variational Autoencoders, mixture-models, expectation-maximization procedure.
- George M. Church (en af forfatterne til den der atikel Wouter sendte til os) er et bigshot inden for protein something.

### Muligheder for projekt og speciale
- Projekt uden for kursusregi: Kig på artiklen og se om vi kan reproducere noget af det, forstå det osv.
- Speciale: (1) Kan vi lære representationer for de der proteiner i latent space. (2) Mere teoretisk, geometrien af det latente rum er ikke Euclidean, så når man skal finde veje i det latente rum skal man bruge geodesics og ikke lineær interpolering. Varians er mindst på de der geodesics-veje, mener jeg Wouter sagde.
- BLAST-algoritme. Er nice på proteiner der folder, men ikke på 'disordered proteins'; når man blaster disordered proteins går det galt fordi de ikke folde som **normale** proteiner, så de er svære at finde med blast. Vi kan lave en blast 2.0 med den nye teknik i artiklen, dvs. gennem nearest neighbor is latent rum.
- Something, noget med Novozymes og deres vaskepulver. Enzymer skal være stabile så det bruger man meget tid på. Det kunne man også kigge på.

## 2019-06-25: Log fra møde på Panum før sommerferie med Guillermo, Wouter og Pablo.
- Guillermo har sendt en masse materiale som vi kan kigge på for at få en bedre ide om wtf det er for noget de har gang i: 
  1. [TUTORIAL](https://cryosparc.com/docs/tutorials/3d-variability-analysis/)
  2. [RELION](https://www3.mrc-lmb.cam.ac.uk/relion/index.php?title=Main_Page)
  3. [CISTEM](https://cistem.org/)
  4. [cryoSPARC](https://cryosparc.com/)
  5. Reviews: [REVIEW 1](https://www.sciencedirect.com/science/article/pii/S1369527417301315) [REVIEW 2](https://www.sciencedirect.com/science/article/pii/S0076687916300271)

- Guillermo sagde at der var to pointer:
  (1) People have tried to use ML to not pick up many particles .. ??? Problem: Detector cannot easily identify difficult particles.
  (2) Dunno.

- Random noter fra mødet: data collection -> image processing. Discussion: particle picking. Signal accumulate; where is the signal from particles. They use 2D for cleaning, then 3D for preferential orientation, can fix computationally. Alternate conformations. "topaz" -> Laplacian of Gaussian way to do it. Refinement is semi-automatic. Fourier-space. They need a lot of cleaning after picking, but before modelling. Guillermo wants a way to assess 2 - 4 conformations given a data set. MBR data collection: People use those to compare. Guillermo have a shit ton of data we can use: Robisome data with difficult/complex conformations. Crystal structure, they do a lot of cleaning, removing things. The lens of the microscope creates abberations in the images/data. Project: Check how proteins work, preproject: get better understanding, try to reproduce results.
