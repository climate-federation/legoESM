---
active: true
iteration: 2
session_id: 
max_iterations: 0
completion_promise: "DONE"
started_at: "2026-04-08T12:01:46Z"
---

We still have major issues in the cubed-sphere implementation. The cosine bell case is now reasonable at least up to 1    
  day. But we still have some serious issues with Williamson cases 2 and 5: in Williamson case 2 for instance the u wind should be homogeneous in   
                                                                                                                                                    
  the horizontal and the v wind should be near zero. Do not improvise, follow exactly the FV3 implementation                                        
                                                                                                                                                    
  https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere, https://www.sciencedirect.com/science/article/pii/S0021999124008660,                        
                                                                                                                                                    
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003712, https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004430,     
                                                                                                                                                    
  until the wind field is perfect. Do not accept approximate solution or results, be very accurate. Always investigate the outputs and analyze      
                                                                                                                                                    
  visually . Write the summary of your efforts in FV3_implementation.md so we can keep track of them --max-terations 200
