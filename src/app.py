import streamlit as st
import pandas as pd
import joblib
import os

from pathlib import Path
from datetime import date, datetime

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="",
    layout="wide")

@st.cache_resource
def cargar_modelo():
    '''
    Cargar el modelo ya hecho en el train.py
    '''
    try:
        MODELO_FILE = Path(__file__).parent.parent / "models" / "EXP_01.pkl"

        if not MODELO_FILE.exists():
            st.error(f'Modelo no encontrado en {MODELO_FILE}')
            return None
        
        with st.spinner('Cargando modelo...'):
            modelo = joblib.load(MODELO_FILE)
            st.success('Modelo cargado  ')
            return modelo
    except Exception as e:
        st.error(f'Error al cargar el modelo: {str(e)}')
        return None
modelo = cargar_modelo()

def alimentar_pipeline(datos_usuario):
    '''
    Convertir la información del formulario a una entrada que si acepte el pipeline
    Args:
        datos_usuario (dict): Todos los valores que venian en el formulario
    Outputs:
        df (pd.DataFrame): DataFrame con los datos del usuario
    '''
    df = pd.DataFrame([datos_usuario], index=None)
    return df

###########################################################################################################
# Intergaz con Streamlit
###########################################################################################################
st.title('Churn para clientes en el sector medico')

c1, c2 = st.columns([4, 2])

with c1:
    st.header('Ingrese los datos del cliente')
    cols = st.columns(3)
    with cols[0]:
        # Age	                     - numerico
        age = st.number_input('***Edad***', min_value=0, max_value=120, value=30, step=1)

        # Gender	                 - Binario
        gender = st.radio("***Género***",["Masculino", "Femenino"], index=None)

        # State	                     - opcion desplegable
        state = st.selectbox('***Estado***', options=['NC', 'IL', 'FL', 'PA', 'NY', 'OH', 'CA', 'GA', 'TX', 'MI'])
        
        # Tenure_Months	             - numérico
        tenure_months = st.slider('***Meses de permanencia***',
         min_value=0,
        max_value=120,  # 10 años máximo
        value=12,
        step=1,
        help="Arrastra para seleccionar los meses de antigüedad")

        # Specialty	                 - opcion desplegable
        specialty = st.selectbox('***Especialidad***', options=['Médico General', 
                                                          'Medicina Familiar',
                                                          'Ortopedista',
                                                          'Neurología',
                                                          'Pediatra',
                                                          'Internista',
                                                          'Cardiología'])
        
        # Insurance_Type             - opción desplegable
        insurance_type = st.selectbox('***Tipo de pago***', options = ['Self-Pay', 
                                                                       'Medicare', 
                                                                       'Privado', 
                                                                       'Medicaid'])
        # Visits_Last_Year	         - numérico
        visits_last_year = st.number_input('***Visitas en el último año***', min_value = 0, max_value = 50, value = 5, step = 1)

       
    with cols[1]:

             # Missed_Appointments	     - numérico
        missed_appointments = st.number_input('***Citas perdidas***', min_value=0, max_value=10, value=5, step=1)
        
        # Days_Since_Last_Visit	     - numérico
        days_since_last_visit = st.number_input('***Días desde la última visita***', min_value=0, max_value=15, value=0, step=1)
    

        # Last_Interaction_Date      - fecha
        last_interaction_date = st.date_input(
            "***Última fecha de interacción***",
            value=date.today(), 
            min_value=date(2020, 1, 1),  
            max_value=date(2026, 12, 31),  
            format="YYYY/MM/DD", 
            help="Selecciona la última fecha que el paciente interactuó con el sistema")
        st.write(f"Fecha seleccionada: {last_interaction_date}")

        # Overall_Satisfaction	     - ranking float
        overall_satisfaction = st.slider('***Satisfacción general***',
        min_value=0.0,
        max_value=5.0,  
        value=2.5,
        step=0.1,
        help="Arrastra para seleccionar")


        # Wait_Time_Satisfaction     - ranking float
        wait_time_satisfaction = st.slider('***Satisfacción de tiempo de espera***',
        min_value=0.0,
        max_value=5.0,  
        value=2.5,
        step=0.1,
        help="Arrastra para seleccionar")

        # Staff_Satisfaction	     - ranking float
        staff_satisfaction = st.slider('***Satisfacción del personal***',
        min_value=0.0,
        max_value=5.0,  
        value=2.5,
        step=0.1,
        help="Arrastra para seleccionar")

    with cols[2]:
         # Provider_Rating	         - ranking float
        provider_rating = st.slider('***Satisfacción del proveedor***',
        min_value=0.0,
        max_value=5.0,  
        value=2.5,
        step=0.1,
        help="Arrastra para seleccionar")

        # Avg_Out_Of_Pocket_Cost     - numérico
        avg_out_of_pocket_cost = st.slider('***Costo promedio de consultas***',
        min_value=0,
        max_value=2500,
        value=1000,
        step=1,
        help="Arrastra para seleccionar los meses de antigüedad")

        # Billing_Issues	         - numérico
        billing_issues = st.radio('***¿Problemas de facturación?***', options=['Si', 'No'], horizontal=True)

        # Portal_Usage	             - numérico
        portal_usage = st.radio("***Hace uso de el portal web?***",["Si", "No"], index=None, horizontal = True)
        
        # Referrals_Made             - numérico
        referrals_made = st.number_input('***Usuarios referidos***', min_value=0, max_value=5, value=1, step=1)

        # Distance_To_Facility_Miles - numperico flotante
        distance_to_facility_miles = st.slider('***Distancia a la clínica más cercana en Millas***',
        min_value=0,
        max_value=50,
        value=25,
        step=1,
        help="Arrastra para seleccionar")

with c2:
    st.subheader("Predicción del modelo")
    
    umbral = st.slider(
        '***Umbral de decisión para Churn***',
        min_value=10,
        max_value=90,
        value=50,
        step=5,
        format="%.0f%%",
        help="Bajarlo haría la predicción mas sensible y aumentaría el numero de Churns, subirlo haría la predicción mas confiable pero habría mas Falsos Negativos")
    
    if modelo is not None:
        try:
            datos_usuario = { 'Patient_Id':None,
                              'Age': age,
                              'Gender': 'Male' if gender == 'Masculino' else 'Female',
                              'State': state,
                              'Tenure_Months': tenure_months,
                              'Specialty': specialty,
                              'Insurance_Type': insurance_type,
                              'Visits_Last_Year': visits_last_year,
                              'Missed_Appointments': missed_appointments,
                              'Days_Since_Last_Visit': days_since_last_visit,
                              'Last_Interaction_Date': last_interaction_date,
                              'Overall_Satisfaction': overall_satisfaction,
                              'Wait_Time_Satisfaction': wait_time_satisfaction,
                              'Staff_Satisfaction': staff_satisfaction,
                              'Provider_Rating': provider_rating,
                              'Avg_Out_Of_Pocket_Cost': avg_out_of_pocket_cost,
                              'Billing_Issues': 1 if billing_issues == 'Si' else 0,
                              'Portal_Usage': 1 if portal_usage == 'Si' else 0,
                              'Referrals_Made': referrals_made,
                              'Distance_To_Facility_Miles': distance_to_facility_miles}
            df = alimentar_pipeline(datos_usuario)

            # Predicción
            if hasattr(modelo, 'predict_proba'):
                probabilidad = modelo.predict_proba(df)[0][1]
                pred = 1 if probabilidad >= (umbral/100) else 0

                # Tabla 
                tabla_resultados = pd.DataFrame({
                    'Probabilidad No Churn': [f"{1-probabilidad:.1%}"],
                    'Probabilidad Churn': [f"{probabilidad:.1%}"],
                    'Predicción': ['Churn' if pred == 1 else 'No Churn']
                })
                st.dataframe(tabla_resultados, 
                             hide_index = True)

            else:
                pred = modelo.predict(df)[0]
                probabilidad = None

            # Mostrar resultados
            if pred == 1:
                st.error('Potencial riesgo de Churn con el usuario')
            else:
                st.success('No existe un riesgo de Churn actualmente')

        except Exception as e:
            st.error(f'Error al mostrar resultados: {str(e)}')